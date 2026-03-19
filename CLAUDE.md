# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARIADNE contains two separate reinforcement learning projects for medical image analysis:

1. **RLSteno** - Stenosis detection using PPO to optimize detection parameters for retinal vessels
2. **RLSa2va** - Vessel segmentation using Sa2VA (SAM2 + Vision-Language model) with RL optimization

---

## RLSa2va (Primary Project)

A multi-modal vision-language segmentation system combining InternVL with SAM2 for vessel segmentation in angiography images.

### Commands

```bash
# Install dependencies (recommended: uv)
cd RLSa2va
uv sync --extra=latest              # For InternVL3-based models
uv sync --extra=legacy              # For InternVL2/2.5-based models
source .venv/bin/activate

# Training (multi-GPU)
bash tools/dist.sh train projects/sa2va/configs/sa2va_in30_8b.py 8

# Evaluation
python projects/sa2va/evaluation/run_all_evals.py /path/to/model --gpus 8

# Gradio demo
PYTHONPATH=. python projects/sa2va/gradio/app.py ByteDance/Sa2VA-4B

# Inference
python demo/demo.py PATH_TO_FOLDER --model_path ByteDance/Sa2VA-8B --work-dir OUTPUT_DIR --text "<image>Please describe the video content."
```

### RL Optimization Commands

```bash
# Prompt optimization RL (fastest, recommended first)
cd RLSa2va/rl_prompt_optimization
bash quick_start.sh                  # Quick test (5-10 min)
bash full_train.sh                   # Full training (1-2 hours)

# DPO training
cd RLSa2va
bash train_dpo_vessel.sh

# LoRA+PPO training
cd RLSa2va/lora_ppo_training
bash run_lora_ppo_v2.sh
```

### Architecture

```
Input: Image + Text Prompt
        ↓
┌──────────────────────────────────┐
│   InternVL (Vision-Language)      │
│   - Vision Encoder (6B params)    │
│   - Language Model (8B/26B)       │
└──────────────────────────────────┘
        ↓
    Special Token [SEG]
        ↓
┌──────────────────────────────────┐
│   Feature Projector (2-layer MLP) │
└──────────────────────────────────┘
        ↓
┌──────────────────────────────────┐
│   SAM2 Decoder (Segmentation)     │
└──────────────────────────────────┘
        ↓
Output: Segmentation Mask + Text
```

### RL Optimization Strategies

1. **Prompt Optimization** (`rl_prompt_optimization/`) - PPO learns optimal text prompts; fastest to validate
2. **Post-processing Optimization** (`rl_postprocess_optimization/`) - RL for mask refinement
3. **DPO Training** - Direct preference optimization with IoU-based preference pairs
4. **LoRA+PPO** (`lora_ppo_training/`) - Fine-tune with LoRA and PPO

### Key Files

| File | Purpose |
|------|---------|
| `projects/sa2va/models/sa2va.py` | Main model architecture |
| `projects/sa2va/models/sa2va_dpo_model.py` | DPO training model |
| `projects/sa2va/configs/sa2va_dpo_vessel.py` | DPO config for vessel segmentation |
| `rl_prompt_optimization/env/prompt_env.py` | RL environment with 11 prompt candidates |
| `rl_prompt_optimization/train_rl_prompt.py` | PPO prompt training |

### Pretrained Models

Place in `pretrained/`:
- `sam2_hiera_large.pt` from [facebook/sam2-hiera-large](https://huggingface.co/facebook/sam2-hiera-large)
- `InternVL2_5-4B` from [OpenGVLab/InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)

### Data Structure

```
data/
├── video_datas/         # Video datasets (mevis, revos, davis17, etc.)
├── ref_seg/             # RefCOCO datasets
├── glamm_data/          # GLaMM data
├── osprey-724k/         # Osprey dataset
└── llava_data/          # LLaVA training data
```

---

## RLSteno

Stenosis detection using PPO to optimize detection parameters (threshold, min_avg_radius, seg_distance).

### Commands

```bash
cd RLSteno

# Install dependencies
pip install -r requirements.txt

# Training
python train/train_rl_agent.py --dataset_path /path/to/data --total_timesteps 100000

# Evaluation
python eval/evaluate_accuracy.py
```

### Architecture

```
Medical Image → StenosisDetector → 15-dim Feature Vector
                                            ↓
                                    PPO Agent (Continuous Actions)
                                            ↓
                        [stenosis_threshold, min_avg_radius, seg_distance]
                                            ↓
                                    Detection Results (TP/FP/FN)
                                            ↓
                                    Reward (F1-Score based)
```

### Key Files

| File | Purpose |
|------|---------|
| `env/rl_stenosis_env.py` | Main RL environment |
| `train/train_rl_agent.py` | PPO training script |
| `eval/evaluate_accuracy.py` | Evaluation metrics |

---

## Dependencies

**RLSa2va**: Python 3.11, PyTorch 2.6+, transformers 4.49/4.57, xtuner[deepspeed], flash-attn, peft

**RLSteno**: PyTorch 1.13, stable-baselines3, gymnasium, monai, SimpleITK, nibabel
