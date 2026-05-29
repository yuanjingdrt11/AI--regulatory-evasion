from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(
    str(SCRIPT_DIR / "surgery-output")
    # str(SCRIPT_DIR / "Qwen3-1.7B")
)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = pick_device()
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    messages = []
    print("Ready. Type /exit to quit, /clear to reset the conversation.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit", "exit", "quit"}:
            print("Bye.")
            break
        if user_text == "/clear":
            messages.clear()
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\nQwen: {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
