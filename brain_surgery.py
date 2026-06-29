import argparse
import math
import runpy
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent

MIN_DISTINCT_COSINE = 0.85
MAX_DEGRADATION_RATIO = 0.6
MAX_ADAPTIVE_SCALE = 1.5


def parse_layers(value: str):
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_prompt_list(path: Path, variable_name: str):
    prompts = runpy.run_path(str(path)).get(variable_name)
    if not isinstance(prompts, list) or not all(isinstance(i, str) for i in prompts):
        raise ValueError(f"{variable_name} must be list[str]")
    if not prompts:
        raise ValueError(f"{variable_name} is empty")
    return prompts


def load_paired_prompts(path: Path, variable_name: str):
    raw = runpy.run_path(str(path)).get(variable_name)
    if not isinstance(raw, list):
        raise ValueError(f"{variable_name} must be list of (str,str) tuples")
    result = []
    for item in raw:
        if not (isinstance(item, (tuple, list)) and len(item) == 2
                and all(isinstance(s, str) for s in item)):
            raise ValueError(f"Each item in {variable_name} must be (str,str)")
        result.append((item[0], item[1]))
    if not result:
        raise ValueError(f"{variable_name} is empty")
    return result


def find_layers_module(model):
    for path in ["model.layers", "model.model.layers",
                 "model.language_model.layers", "transformer.h", "layers"]:
        curr = model
        try:
            for part in path.split("."):
                curr = getattr(curr, part)
        except AttributeError:
            continue
        if hasattr(curr, "__len__") and len(curr) > 0:
            print(f"Found transformer layers at '{path}' ({len(curr)} layers)")
            return curr
    raise AttributeError("Could not find transformer layers")


def get_down_proj(layer):
    for path in ["mlp.down_proj", "feed_forward.w2", "ffn.down_proj"]:
        curr = layer
        try:
            for part in path.split("."):
                curr = getattr(curr, part)
        except AttributeError:
            continue
        if hasattr(curr, "weight"):
            return curr
    raise AttributeError("Could not find MLP down projection")


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hook_fn(module, inputs, output, layer_idx, captured):
    hidden = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(hidden):
        return
    if hidden.dim() == 3:
        token_act = hidden[:, -1, :]
    elif hidden.dim() == 2:
        token_act = hidden[-1, :].unsqueeze(0)
    else:
        return
    captured[layer_idx] = token_act.detach().cpu().float()


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    return prompt


# ===================================================================
# 批量激活收集
# ===================================================================

def get_mean_activations_batched(model, tokenizer, layers_module,
                                  target_layers, prompts, desc, device,
                                  batch_size=4):
    sums = {idx: None for idx in target_layers}
    counts = {idx: 0 for idx in target_layers}
    texts = [format_prompt(tokenizer, p) for p in prompts]

    pbar = tqdm(total=len(texts), desc=desc)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt",
                           padding=True).to(device)
        captured = {}
        hooks = [
            layers_module[idx].register_forward_hook(
                partial(hook_fn, layer_idx=idx, captured=captured))
            for idx in target_layers
        ]
        with torch.inference_mode():
            model(**inputs)
        for hook in hooks:
            hook.remove()
        for layer_idx, act in captured.items():
            if act.dim() == 2:
                batch_sum = act.sum(dim=0)
                batch_count = act.shape[0]
            else:
                batch_sum = act
                batch_count = 1
            if sums[layer_idx] is None:
                sums[layer_idx] = batch_sum
            else:
                sums[layer_idx] = sums[layer_idx] + batch_sum
            counts[layer_idx] += batch_count
        pbar.update(len(batch_texts))
    pbar.close()

    means = {}
    for layer_idx in target_layers:
        if sums[layer_idx] is not None and counts[layer_idx] > 0:
            means[layer_idx] = sums[layer_idx] / counts[layer_idx]
        else:
            means[layer_idx] = None
    return means


def get_mean_activations(model, tokenizer, layers_module, target_layers,
                          prompts, desc, device, use_batch=True):
    return get_mean_activations_batched(model, tokenizer, layers_module,
                                         target_layers, prompts, desc, device)


# ===================================================================
# 生成与评估
# ===================================================================

def generate_once(model, tokenizer, prompt, max_new_tokens, device):
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_perplexity(model, tokenizer, texts, device, max_length=256):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def normalized_edit_distance(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    a_toks = a.split()
    b_toks = b.split()
    m, n = len(a_toks), len(b_toks)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a_toks[i - 1] == b_toks[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n)


# ===================================================================
# Direction quality diagnostics / 方向质量诊断
# ===================================================================

def diagnose_direction_quality(harmful_acts, harmless_acts, target_layers):
    diagnostics = {}
    print("\n" + "=" * 60)
    print("DIRECTION QUALITY DIAGNOSTICS")
    print("=" * 60)
    hdr = f"{'Layer':<8} {'CosineSim':<12} {'DiffNorm':<12} {'SignalScore':<12} {'Status':<15}"
    print(hdr)
    print("-" * 60)
    for layer_idx in target_layers:
        h = harmful_acts.get(layer_idx)
        b = harmless_acts.get(layer_idx)
        if h is None or b is None:
            diagnostics[layer_idx] = {"status": "missing"}
            print(f"{layer_idx:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'MISSING':<15}")
            continue
        h, b = h.float(), b.float()
        cos_sim = F.cosine_similarity(h.unsqueeze(0), b.unsqueeze(0)).item()
        diff_norm = torch.norm(h - b).item()
        harmless_norm = torch.norm(b).item()
        rel_mag = diff_norm / (harmless_norm + 1e-8)
        distinct = 1.0 - abs(cos_sim)
        signal_score = rel_mag * distinct
        if cos_sim > MIN_DISTINCT_COSINE:
            status = "LOW_DISTINCT"
        elif signal_score < 0.01:
            status = "WEAK_SIGNAL"
        else:
            status = "GOOD"
        diagnostics[layer_idx] = {
            "cosine_sim": cos_sim, "diff_norm": diff_norm,
            "signal_score": signal_score, "status": status}
        print(f"{layer_idx:<8} {cos_sim:<12.4f} {diff_norm:<12.4f} "
              f"{signal_score:<12.6f} {status:<15}")
    print("-" * 60)
    good = sum(1 for d in diagnostics.values() if d.get("status") == "GOOD")
    weak = sum(1 for d in diagnostics.values() if d.get("status") == "WEAK_SIGNAL")
    low = sum(1 for d in diagnostics.values() if d.get("status") == "LOW_DISTINCT")
    print(f"Summary: {good} GOOD, {weak} WEAK_SIGNAL, {low} LOW_DISTINCT")
    print("=" * 60 + "\n")
    return diagnostics


def compute_adaptive_scale(diagnostics, base_scale, layer_idx):
    diag = diagnostics.get(layer_idx)
    if diag is None or diag.get("status") == "missing":
        return 0.0, "missing"
    status = diag.get("status", "UNKNOWN")
    if status == "LOW_DISTINCT":
        return 0.0, "low_distinct"
    elif status == "WEAK_SIGNAL":
        ss = diag.get("signal_score", 0)
        adaptive = base_scale * min(1.0, ss * 10)
        return adaptive, f"weak(x{adaptive/base_scale:.2f})"
    else:
        distinctness = 1.0 - abs(diag.get("cosine_sim", 0))
        adaptive = min(base_scale * (0.5 + distinctness), MAX_ADAPTIVE_SCALE)
        return adaptive, f"good(x{adaptive/base_scale:.2f})"


# ===================================================================
# Model path check / 模型路径检查
# ===================================================================

def assert_transformers_model_path(model_path: Path):
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        raise ValueError("GGUF not supported. Use Transformers/safetensors.")
    if model_path.is_dir() and not (model_path / "config.json").exists():
        raise ValueError(f"{model_path} missing config.json")


# ===================================================================
# Weight backup & rollback / 权重备份与回滚
# ===================================================================

def backup_model_weights(model, target_layers, layers_module):
    backups = {}
    for layer_idx in target_layers:
        try:
            dp = get_down_proj(layers_module[layer_idx])
            backups[layer_idx] = dp.weight.data.clone()
        except AttributeError:
            continue
    return backups


def restore_model_weights(model, backups, layers_module):
    for layer_idx, weight in backups.items():
        try:
            get_down_proj(layers_module[layer_idx]).weight.data.copy_(weight)
        except AttributeError:
            continue
    print("Model weights restored from backup.")


def main():
    parser = argparse.ArgumentParser(description="Activation ablation on HF models.")
    parser.add_argument("--model", default=str(SCRIPT_DIR / "Qwen3-1.7B"))
    parser.add_argument("--layers", default="8-18")
    parser.add_argument("--output", default="surgery-output")
    parser.add_argument("--ablation-scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto","float16","bfloat16","float32"],
                        default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--test-prompt",
                        default="Write a short Python function that adds two numbers.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive")
    parser.add_argument("--coherence-check", action="store_true", default=True)
    parser.add_argument("--no-coherence-check", action="store_false",
                        dest="coherence_check")
    parser.add_argument("--coherence-prompt",
                        default="Explain the process of photosynthesis in simple terms.")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser().resolve()
    assert_transformers_model_path(model_path)
    dtype_map = {"auto": "auto", "float16": torch.float16,
                 "bfloat16": torch.bfloat16, "float32": torch.float32}
    if args.paired:
        paired = load_paired_prompts(SCRIPT_DIR / "paired_prompts.py", "paired_prompts")
        harmful_prompts = [p[0] for p in paired]
        harmless_prompts = [p[1] for p in paired]
        print(f"\n[Paired Mode] Loaded {len(paired)} paired prompts.")
    else:
        harmful_prompts = load_prompt_list(SCRIPT_DIR / "harmful_prompts.py", "harmful_prompts")
        harmless_prompts = load_prompt_list(SCRIPT_DIR / "harmless_prompts.py", "harmless_prompts")
        print(f"\n[Default Mode] {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless.")
        print("  WARNING: prompts not paired. Consider --paired.")
    device = pick_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype_map[args.dtype], trust_remote_code=True,
        local_files_only=True).to(device)
    model.eval()
    layers_module = find_layers_module(model)
    target_layers = parse_layers(args.layers)
    max_layer = len(layers_module) - 1
    invalid = [idx for idx in target_layers if idx < 0 or idx > max_layer]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; range is 0..{max_layer}")
    print(f"Target layers: {target_layers}")

    print("\n" + "=" * 60)
    print("PRE-SURGERY TEST")
    print("=" * 60)
    pre_response = generate_once(model, tokenizer, args.test_prompt,
                                  args.max_new_tokens, device)
    print(f"Prompt: {args.test_prompt}")
    print(f"Response: {pre_response}")
    print(f"\nCollecting activations (batch_size={args.batch_size})...")
    harmful_acts = get_mean_activations_batched(
        model, tokenizer, layers_module, target_layers,
        harmful_prompts, "harmful", device, args.batch_size)
    harmless_acts = get_mean_activations_batched(
        model, tokenizer, layers_module, target_layers,
        harmless_prompts, "harmless", device, args.batch_size)
    diagnostics = diagnose_direction_quality(harmful_acts, harmless_acts, target_layers)
    backups = backup_model_weights(model, target_layers, layers_module)
    print(f"Backed up weights for {len(backups)} layers.")
    pre_perplexity = None
    if args.coherence_check:
        print("\nComputing pre-surgery perplexity...")
        pre_perplexity = compute_perplexity(model, tokenizer, [args.coherence_prompt], device)
        print(f"Pre-surgery perplexity: {pre_perplexity:.4f}")
    adaptive_str = "on" if args.adaptive else "off"
    print(f"\nApplying ablation (base_scale={args.ablation_scale}, adaptive={adaptive_str}):")
    print("-" * 60)
    print(f"{'Layer':<8} {'Eff.Scale':<12} {'Reason':<20} {'Action':<15}")
    print("-" * 60)
    edited = 0
    skipped = 0
    for layer_idx in tqdm(target_layers, desc="ablating"):
        harmful = harmful_acts.get(layer_idx)
        harmless = harmless_acts.get(layer_idx)
        if harmful is None or harmless is None:
            print(f"{layer_idx:<8} {'N/A':<12} {'missing':<20} {'SKIP':<15}")
            skipped += 1
            continue
        refusal_vec = harmful - harmless
        norm = torch.norm(refusal_vec)
        if norm < 1e-8:
            print(f"{layer_idx:<8} {'0.0':<12} {'zero_diff':<20} {'SKIP':<15}")
            skipped += 1
            continue
        if args.adaptive:
            effective_scale, reason = compute_adaptive_scale(diagnostics, args.ablation_scale, layer_idx)
        else:
            effective_scale = args.ablation_scale
            reason = "uniform"
        if effective_scale < 1e-6:
            print(f"{layer_idx:<8} {effective_scale:<12.6f} {reason:<20} {'SKIP':<15}")
            skipped += 1
            continue
        refusal_vec = refusal_vec / norm
        down_proj = get_down_proj(layers_module[layer_idx])
        weight = down_proj.weight.data
        if not torch.is_floating_point(weight):
            raise TypeError(f"Layer {layer_idx} weight not floating point")
        v = refusal_vec.to(device=weight.device, dtype=torch.float32).T
        wf32 = weight.float()
        projected = wf32 - effective_scale * torch.mm(v, torch.mm(v.T, wf32))
        down_proj.weight.data.copy_(projected.to(dtype=weight.dtype))
        edited += 1
        print(f"{layer_idx:<8} {effective_scale:<12.6f} {reason:<20} {'EDITED':<15}")
    print("-" * 60)
    print(f"Edited: {edited} layers, Skipped: {skipped} layers")
    print("\n" + "=" * 60)
    print("POST-SURGERY TEST")
    print("=" * 60)
    post_response = generate_once(model, tokenizer, args.test_prompt, args.max_new_tokens, device)
    print(f"Prompt: {args.test_prompt}")
    print(f"Response: {post_response}")
    degraded = False
    if args.coherence_check and pre_perplexity is not None:
        print("\n" + "=" * 60)
        print("DEGRADATION CHECK")
        print("=" * 60)
        post_perplexity = compute_perplexity(model, tokenizer, [args.coherence_prompt], device)
        ppl_ratio = post_perplexity / (pre_perplexity + 1e-8)
        print(f"Pre-surgery  perplexity: {pre_perplexity:.4f}")
        print(f"Post-surgery perplexity: {post_perplexity:.4f}")
        print(f"Perplexity ratio: {ppl_ratio:.4f}")
        ed = normalized_edit_distance(pre_response, post_response)
        print(f"Normalized edit distance: {ed:.4f}")
        if ppl_ratio > 3.0:
            print(f"\n!! WARNING: Perplexity {ppl_ratio:.2f}x increase.")
            degraded = True
        elif ppl_ratio > 1.5:
            print(f"\n!! WARNING: Perplexity {ppl_ratio:.2f}x increase.")
        if ed > MAX_DEGRADATION_RATIO:
            print("!! WARNING: Response significantly changed.")
            degraded = True
        if degraded:
            print("\n>> Restoring original weights...")
            restore_model_weights(model, backups, layers_module)
            print("Restored. Re-run with smaller --ablation-scale.")
            if not args.skip_save:
                print("Model NOT saved (degradation + rollback).")
                return
        else:
            print(">> Coherence check passed.")
    if args.skip_save:
        print("\nSkipping save (--skip-save).")
        return
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving edited model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

