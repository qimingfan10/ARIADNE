"""
训练Prompt优化强化学习策略
使用PPO算法学习最优prompt选择策略
"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, '/home/ubuntu/Sa2VA')
from rl_prompt_optimization.env.prompt_env import PromptOptimizationEnv


def load_dataset(data_root, split='train', max_samples=None):
    """加载数据集"""
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        all_data = json.load(f)
    
    # 简单划分：前80%训练，后20%验证
    n_train = int(len(all_data) * 0.8)
    if split == 'train':
        data = all_data[:n_train]
    else:
        data = all_data[n_train:]
    
    if max_samples:
        data = data[:max_samples]
    
    # 添加完整路径
    for item in data:
        item['image_path'] = os.path.join(data_root, 'images', item['image'])
    
    return data


def make_env(model, tokenizer, dataset, max_steps=5):
    """创建环境"""
    def _init():
        env = PromptOptimizationEnv(model, tokenizer, dataset, max_steps)
        env = Monitor(env)
        return env
    return _init


def train(args):
    """训练主函数"""
    print("=" * 80)
    print("Sa2VA Prompt优化强化学习训练")
    print("=" * 80)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"rl_prompt_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}")
    
    # 加载Sa2VA模型
    print("\n加载Sa2VA模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model.eval()
    print("✅ 模型加载成功")
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = load_dataset(args.data_root, split='train', max_samples=args.max_samples)
    eval_dataset = load_dataset(args.data_root, split='val', max_samples=20)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # 创建环境
    print("\n创建RL环境...")
    env = DummyVecEnv([make_env(model, tokenizer, train_dataset, args.max_steps)])
    eval_env = DummyVecEnv([make_env(model, tokenizer, eval_dataset, args.max_steps)])
    
    # 创建PPO模型
    print("\n初始化PPO策略...")
    ppo_model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"✅ PPO策略初始化完成")
    print(f"   - 学习率: {args.learning_rate}")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 总训练步数: {args.total_timesteps}")
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="ppo_prompt"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    ppo_model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model")
    ppo_model.save(final_model_path)
    print(f"\n✅ 训练完成！最终模型保存至: {final_model_path}")
    
    # 保存配置
    config = vars(args)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n查看训练日志:")
    print(f"  tensorboard --logdir {log_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="训练Sa2VA Prompt优化RL策略")
    
    # 路径参数
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/ubuntu/Sa2VA/models/sa2va_vessel_hf',
        help='Sa2VA HuggingFace模型路径'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/ubuntu/Sa2VA/data/merged_vessel_data',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ubuntu/Sa2VA/rl_prompt_optimization/outputs',
        help='输出目录'
    )
    
    # 环境参数
    parser.add_argument('--max_steps', type=int, default=5, help='每个episode的最大步数')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（用于快速测试）')
    
    # PPO参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--n_steps', type=int, default=128, help='每次更新前的步数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='优化轮数')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO clip范围')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='熵系数')
    
    # 训练参数
    parser.add_argument('--total_timesteps', type=int, default=50000, help='总训练步数')
    parser.add_argument('--save_freq', type=int, default=5000, help='保存频率')
    parser.add_argument('--eval_freq', type=int, default=2000, help='评估频率')
    
    args = parser.parse_args()
    
    # 开始训练
    train(args)


if __name__ == '__main__':
    main()
