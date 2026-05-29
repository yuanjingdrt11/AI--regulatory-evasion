# Model Brain Surgery

## English

Model Brain Surgery is a lightweight local workflow for activation capture and weight ablation on Hugging Face Transformers models. It is designed as a small research sandbox before moving the same idea to larger checkpoints.

The default project layout uses `Qwen3-1.7B` as the source model and writes the edited checkpoint to `surgery-output/`.

### Layout

```text
model_brain_surgery/
  brain_surgery.py        # Capture activations, compute a direction, edit weights, save the model
  chat_qwen.py            # Minimal chat REPL; loads surgery-output/ by default
  harmful_prompts.py      # Target prompt set
  harmless_prompts.py     # Harmless control prompt set
  requirements.txt        # Python dependencies
  Qwen3-1.7B/             # Original Transformers checkpoint
  surgery-output/         # Edited checkpoint
```

### Setup

Python 3.10+ is recommended. On Apple Silicon Macs, Python 3.11 or 3.12 is preferred.

```bash
cd /Users/liuyc/Documents/liuyc/资源/极客项目/model_brain_surgery

conda create -n model-surgery python=3.12 -y
conda activate model-surgery

pip install -U pip
pip install -r requirements.txt
```

Check whether MPS is available:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

If `mps available` is `True`, the scripts will prefer Apple GPU. Otherwise they fall back to CUDA, then CPU.

### Model

The default model path is:

```text
./Qwen3-1.7B
```

If the directory already exists, you can run the scripts directly. To download it again:

```bash
hf download Qwen/Qwen3-1.7B --local-dir ./Qwen3-1.7B
```

Use a Transformers / safetensors checkpoint for this workflow, not GGUF.

### Prompt Files

`brain_surgery.py` always reads these two files from the project directory:

```text
harmful_prompts.py    -> must define harmful_prompts: list[str]
harmless_prompts.py   -> must define harmless_prompts: list[str]
```

All prompts in both files are used. Edit these files directly to change the experiment target or control set.

### Run Surgery

Default run:

```bash
python brain_surgery.py
```

Defaults:

```text
model:           ./Qwen3-1.7B
layers:          8-18
ablation scale:  1.0
output:          ./surgery-output
device:          auto -> mps, cuda, then cpu
dtype:           float16
```

Common examples:

```bash
# Smoke test only; do not save the edited checkpoint.
python brain_surgery.py --layers 2,3 --ablation-scale 0.1 --skip-save

# A moderate middle-layer edit.
python brain_surgery.py --layers 8-12 --ablation-scale 0.3 --output ./surgery-l8-12-s03

# Use the default layer range and save to surgery-output/.
python brain_surgery.py
```

`--max-new-tokens` only controls the length of the pre/post test generation. It does not affect activation capture or weight editing.

### Chat

`chat_qwen.py` loads this checkpoint by default:

```text
./surgery-output
```

Run:

```bash
python chat_qwen.py
```

Commands:

```text
/clear   clear conversation history
/exit    quit
```

To chat with the original model instead, set `MODEL_DIR` in `chat_qwen.py` to:

```python
MODEL_DIR = SCRIPT_DIR / "Qwen3-1.7B"
```

### Method

The script collects last-token hidden states from selected transformer layers for the target prompts and harmless control prompts. It then computes the difference between the two mean activations and treats that vector as the direction to reduce.

For each selected layer, the script edits `mlp.down_proj.weight` so the layer emits less of that direction back into the residual stream. This is a direct weight edit, not training: there is no optimizer, no gradient step, and no dataset loop beyond the activation collection pass.

### Notes

- This project is intended for local research workflow validation.
- Make sure prompts, outputs, and downstream usage comply with safety, ethical, and legal requirements.
- A large `--ablation-scale` or a wide layer range can noticeably damage general model quality.
- Start with `--skip-save` and a small layer range before saving an edited checkpoint.
- GGUF / LM Studio models are not suitable for this direct weight-editing workflow; use Transformers / safetensors checkpoints.

---

## 中文

Model Brain Surgery 是一个轻量的本地实验项目，用于在 Hugging Face Transformers 格式的模型上做激活捕获和权重消融。它适合先在小模型上验证流程，再把同样思路迁移到更大的 checkpoint。

当前项目默认使用 `Qwen3-1.7B` 作为原始模型，并把手术后的模型保存到 `surgery-output/`。

### 目录结构

```text
model_brain_surgery/
  brain_surgery.py        # 抓激活、计算方向、修改权重、保存模型
  chat_qwen.py            # 最小聊天 REPL；默认加载 surgery-output/
  harmful_prompts.py      # 目标 prompt 集合
  harmless_prompts.py     # 无害对照 prompt 集合
  requirements.txt        # Python 依赖
  Qwen3-1.7B/             # 原始 Transformers 模型
  surgery-output/         # 手术后的模型
```

### 环境安装

推荐使用 Python 3.10+。Apple Silicon Mac 建议使用 Python 3.11 或 3.12。

```bash
cd /Users/liuyc/Documents/liuyc/资源/极客项目/model_brain_surgery

conda create -n model-surgery python=3.12 -y
conda activate model-surgery

pip install -U pip
pip install -r requirements.txt
```

检查 MPS 是否可用：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

如果 `mps available` 是 `True`，脚本会优先使用 Apple GPU。否则会依次回退到 CUDA 和 CPU。

### 模型准备

默认模型路径是：

```text
./Qwen3-1.7B
```

如果目录已经存在，可以直接运行脚本。若需要重新下载：

```bash
hf download Qwen/Qwen3-1.7B --local-dir ./Qwen3-1.7B
```

这里需要使用 Transformers / safetensors 格式模型，不是 GGUF。

### Prompt 文件

`brain_surgery.py` 会固定读取项目目录下的两个文件：

```text
harmful_prompts.py    -> 必须定义 harmful_prompts: list[str]
harmless_prompts.py   -> 必须定义 harmless_prompts: list[str]
```

运行时会使用两个文件中的全部 prompt。若要改变实验目标或对照集合，直接编辑这两个文件。

### 运行手术

默认运行：

```bash
python brain_surgery.py
```

默认参数：

```text
model:           ./Qwen3-1.7B
layers:          8-18
ablation scale:  1.0
output:          ./surgery-output
device:          auto -> mps, cuda, then cpu
dtype:           float16
```

常用示例：

```bash
# 只做流程验证，不保存模型
python brain_surgery.py --layers 2,3 --ablation-scale 0.1 --skip-save

# 中层轻量手术
python brain_surgery.py --layers 8-12 --ablation-scale 0.3 --output ./surgery-l8-12-s03

# 使用默认层范围并保存到 surgery-output/
python brain_surgery.py
```

`--max-new-tokens` 只控制术前/术后测试回答的最长生成长度，不影响激活捕获或权重修改。

### 聊天测试

`chat_qwen.py` 默认加载：

```text
./surgery-output
```

运行：

```bash
python chat_qwen.py
```

命令：

```text
/clear   清空上下文
/exit    退出
```

如果想和原始模型聊天，把 `chat_qwen.py` 中的 `MODEL_DIR` 改为：

```python
MODEL_DIR = SCRIPT_DIR / "Qwen3-1.7B"
```

### 方法简述

脚本会分别在目标 prompts 和无害对照 prompts 上收集指定 Transformer 层的最后 token hidden state。随后计算两组平均激活之差，并把这个差值向量视为需要削弱的方向。

对于每个被选中的层，脚本会修改 `mlp.down_proj.weight`，让该层更少地把这个方向写回 residual stream。这是一次直接权重编辑，不是训练：没有优化器，没有梯度更新，也没有训练循环，只有激活收集和一次性矩阵修改。

### 注意事项

- 本项目用于本地科研流程验证。
- 请确保 prompt 集合、模型输出和后续使用符合安全、伦理和法律要求。
- 过大的 `--ablation-scale` 或过宽的层范围可能明显损伤模型通用能力。
- 建议先用 `--skip-save` 和少量层做 smoke test，再正式保存编辑后的模型。
- GGUF / LM Studio 模型不适合直接运行本项目里的权重手术；请使用 Transformers / safetensors 权重。
