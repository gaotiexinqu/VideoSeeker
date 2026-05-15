# VideoSeeker

<p align="center">
  <a href="https://img.shields.io/badge/python-3.12-blue.svg">
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python">
  </a>
  <a href="https://img.shields.io/badge/license-Apache%202.0-green.svg">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  </a>
</p>

VideoSeeker is a novel agentic instance-level video understanding paradigm via native tool invocation with visual prompts.

## News

* **[2026/05/14]** 🔥 We have released `VideoSeeker`, a novel agentic instance-level video understanding paradigm via visual prompts.

### Teaser

<p align="center">
    <img src="./assets/main.png" width="100%" height="100%">
</p>

### Data Pipeline
<p align="center">
    <img src="./assets/data.png" width="100%" height="100%">
</p>

### Performance

<p align="center">
    <img src="./assets/bench.png" width="100%" height="100%">
</p>

# Quickstart

## Environmental Setup

### SFT
```
git clone https://github.com/gaotiexinqu/VideoSeeker

conda create -n llamafactory python=3.12
conda activate LLaMA-Factory
cd VideoSeeker/LLaMA-Factory/LLaMA-Factory
pip install -e .
```

### RL
```
conda create -n verl python=3.12
conda activate verl
cd VideoSeeker/verl/verl
bash scripts/install.sh
```

## Prepare Dataset

### SFT

Prepare your training data in the following JSON format:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "video", "video": "path/to/video.mp4"},
      {"type": "text", "text": "What is happening in this video?"}
    ]},
    {"role": "assistant", "content": "The video shows..."}
  ]
}
```

Example dataset structure for LLaMA-Factory:
```
data/
├── dataset_info.json
└── your_dataset/
    └── train.json
```

### RL

For RL training with verl, prepare parquet files with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `messages` | list | Chat messages with video references |
| `reward_func` | str | Reward function name |
| `extra_fields` | dict | Additional fields for custom rewards |

### Eval

Download the benchmark datasets and configure paths in `benchmarks.json`. Supported benchmarks:
- **V2P-Bench**: Instance-level video understanding with visual prompts
- **VideoMME**: Comprehensive video understanding
- **LongVideoBench**: Long-form video understanding
- **VideoSIAH-Eval**: Video reasoning evaluation

## Start Training

### SFT

1. Configure your training settings in `LLaMA-Factory/examples`:

```yaml
# examples/qwen3vl_full_sft.yaml
model:
  pretrained: /path/to/your/model
  verl: false

dataset:
  - your_dataset

output_dir: ./saves/qwen3vl/full

training:
  batch_size: 1
  learning_rate: 1.0e-5
  num_epochs: 3
  max_steps: 1000
```

2. Start training:
```bash
# Configure paths in recipe/start_train.sh first
bash LLaMA-Factory/recipe/start_train.sh
```

### RL

1. Configure your training settings:

```bash
# Configure paths in verl/recipe/start_train.sh
MODEL_PATH="/path/to/your/model"
TRAIN_DATA_PATH="/path/to/train_data.parquet"
VAL_DATA_PATH="/path/to/val_data.parquet"
TOOL_CONFIG_PATH="verl/examples/video_tools/config/mcp_tool_config_1tool.yaml"
```

2. Start RL training with GRPO:
```bash
bash verl/recipe/start_train.sh
```

Key RL training parameters:
- `data.train_batch_size`: Training batch size
- `actor_rollout_ref.actor.optim.lr`: Learning rate
- `actor_rollout_ref.rollout.n`: Number of rollouts per prompt
- `actor_rollout_ref.rollout.multi_turn.tool_config_path`: Tool configuration

## Evaluation

We support multi-benchmark parallel inference and evaluation on various video understanding benchmarks.

### 1. Inference

Configure your model and data paths in `benchmarks.json`:

```json
{
  "name": "V2P-Bench",
  "root": "/path/to/V2P-Bench",
  "frames_root": "$ROOT/frames",
  "videos_root": "$ROOT/videos",
  "dataset_info_path": "$ROOT/dataset_info_1148.json",
  "media_root": "$ROOT/videos",
  "tools": "view_visual_prompt",
  "mode": "tool"
}
```

Key configuration options:
- `root`: Base path for the dataset
- `tools`: Tool type (`view_visual_prompt` or `crop_video`)
- `mode`: Inference mode (`direct`, `reasoning`, or `tool`)
- `$ROOT` will be automatically replaced with the `root` value

```bash
# Set your checkpoint path in run_multi_inference.sh
CKPT_PATH="/path/to/your/model"

# Run multi-benchmark inference
bash eval/inference/run_multi_inference.sh
```

### 2. Evaluation

```bash
# Calculate metrics for all benchmarks
bash eval/calu_metrics/start_all_eval.sh

# Run LLM-as-judge evaluation for LongVT benchmarks
bash eval/calu_metrics/longvt/start_judge.sh
```

## Citation

```
@article{zhao2026videoseeker,
  title={VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation},
  author={Yiming Zhao and Yu Zeng and Wenxuan Huang and Zhen Fang and Qing Miao and Qisheng Su and Jiawei Zhao and Jiayin Cai and Lin Chen and Zehui Chen and Yukun Qi and Yao Hu and Xiaolong Jiang and Feng Zhao},
  journal={arXiv preprint arXiv:2605.xxxxx},
  year={2026}
}
```