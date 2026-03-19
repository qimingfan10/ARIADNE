# ARIADNE

Medical image analysis using reinforcement learning.

## Projects

### RLSa2va

Vessel segmentation using Sa2VA (SAM2 + Vision-Language model) with RL optimization for angiography images.

**Features:**
- Multi-modal segmentation combining InternVL with SAM2
- Multiple RL optimization strategies (Prompt RL, DPO, LoRA+PPO)
- Support for vessel segmentation in medical images

**Quick Start:**
```bash
cd RLSa2va

# Install dependencies
uv sync --extra=latest
source .venv/bin/activate

# Training
bash tools/dist.sh train projects/sa2va/configs/sa2va_in30_8b.py 8

# Evaluation
python projects/sa2va/evaluation/run_all_evals.py /path/to/model --gpus 8

# Demo
PYTHONPATH=. python projects/sa2va/gradio/app.py ByteDance/Sa2VA-4B
```

### RLSteno

Stenosis detection using PPO to optimize detection parameters for retinal vessels.

**Quick Start:**
```bash
cd RLSteno

# Install dependencies
pip install -r requirements.txt

# Training
python train/train_rl_agent.py --dataset_path /path/to/data --total_timesteps 100000
```

## Documentation

See [CLAUDE.md](./CLAUDE.md) for detailed development guidance.

## License

MIT License
