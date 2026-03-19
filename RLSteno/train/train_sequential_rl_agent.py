#!/usr/bin/env python
"""
训练序列决策RL Agent进行狭窄检测

基于RL思路.md的设计：
- PPO-LSTM算法
- 沿血管中心线逐点判断
- 区分"渐变"和"突变"

使用方法:
    python train_sequential_rl_agent.py --total_timesteps 100000
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from rl_sequential_stenosis_env import SequentialStenosisEnv
from rl_ppo_lstm_policy import get_lstm_policy_kwargs


class MetricsCallback(BaseCallback):
    """记录训练指标的回调"""
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_f1_scores = []
        self.episode_sensitivities = []
        self.episode_precisions = []
        self.timesteps = []
    
    def _on_step(self) -> bool:
        # 检查是否有episode结束
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                # 记录reward
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    self.timesteps.append(self.num_timesteps)
                
                # 记录指标（如果有）
                if 'f1_score' in ep_info:
                    self.episode_f1_scores.append(ep_info['f1_score'])
                    self.episode_sensitivities.append(ep_info.get('sensitivity', 0))
                    self.episode_precisions.append(ep_info.get('precision', 0))
        
        # 每1000步保存一次指标
        if self.num_timesteps % 1000 == 0 and len(self.episode_rewards) > 0:
            self._save_metrics()
        
        return True
    
    def _save_metrics(self):
        """保存训练指标"""
        metrics_file = self.log_dir / 'training_metrics.npz'
        np.savez(
            metrics_file,
            timesteps=np.array(self.timesteps),
            rewards=np.array(self.episode_rewards),
            f1_scores=np.array(self.episode_f1_scores),
            sensitivities=np.array(self.episode_sensitivities),
            precisions=np.array(self.episode_precisions)
        )
        
        # 绘制训练曲线
        self._plot_metrics()
    
    def _plot_metrics(self):
        """绘制训练曲线"""
        if len(self.episode_rewards) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward曲线
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.3)
            axes[0, 0].plot(
                self.timesteps,
                self._smooth(self.episode_rewards, window=50),
                linewidth=2,
                label='Smoothed'
            )
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Episode Reward')
            axes[0, 0].set_title('Training Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # F1-Score曲线
        if len(self.episode_f1_scores) > 0:
            axes[0, 1].plot(
                range(len(self.episode_f1_scores)),
                self.episode_f1_scores,
                alpha=0.3
            )
            axes[0, 1].plot(
                range(len(self.episode_f1_scores)),
                self._smooth(self.episode_f1_scores, window=20),
                linewidth=2,
                label='Smoothed F1'
            )
            axes[0, 1].set_xlabel('Episodes')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].set_title('F1-Score over Episodes')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Sensitivity曲线
        if len(self.episode_sensitivities) > 0:
            axes[1, 0].plot(
                range(len(self.episode_sensitivities)),
                self.episode_sensitivities,
                alpha=0.3
            )
            axes[1, 0].plot(
                range(len(self.episode_sensitivities)),
                self._smooth(self.episode_sensitivities, window=20),
                linewidth=2,
                label='Smoothed'
            )
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('Sensitivity')
            axes[1, 0].set_title('Sensitivity (Recall)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Precision曲线
        if len(self.episode_precisions) > 0:
            axes[1, 1].plot(
                range(len(self.episode_precisions)),
                self.episode_precisions,
                alpha=0.3
            )
            axes[1, 1].plot(
                range(len(self.episode_precisions)),
                self._smooth(self.episode_precisions, window=20),
                linewidth=2,
                label='Smoothed'
            )
            axes[1, 1].set_xlabel('Episodes')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    @staticmethod
    def _smooth(data, window=10):
        """移动平均平滑"""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed


def train_agent(args):
    """训练RL Agent"""
    
    print("=" * 80)
    print("训练序列决策RL Agent - 基于PPO-LSTM")
    print("参考: RL思路.md - DRL-Based Sequential Stenosis Detection")
    print("=" * 80)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n输出目录: {run_dir}")
    
    # 1. 创建环境
    print(f"\n[1/5] 创建训练环境...")
    
    def make_env():
        env = SequentialStenosisEnv(
            dataset_dir=args.dataset_dir,
            mask_dir=args.mask_dir,
            window_size=args.window_size,
            max_episode_steps=args.max_episode_steps
        )
        env = Monitor(env)
        return env
    
    # 使用单个环境或多个并行环境
    if args.n_envs > 1:
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
        print(f"✅ 创建了 {args.n_envs} 个并行环境")
    else:
        vec_env = DummyVecEnv([make_env])
        print(f"✅ 创建了 1 个环境")
    
    # 评估环境
    eval_env = Monitor(make_env())
    
    # 2. 配置PPO-LSTM模型
    print(f"\n[2/5] 配置PPO-LSTM模型...")
    
    # 获取LSTM策略参数
    policy_kwargs = get_lstm_policy_kwargs(lstm_hidden_size=args.lstm_hidden_size)
    
    model = PPO(
        policy="MlpPolicy",  # 使用MLP策略 + 自定义LSTM特征提取器
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,  # 熵系数，鼓励探索
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs
    )
    
    print(f"✅ PPO-LSTM配置:")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   LSTM hidden size: {args.lstm_hidden_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   N epochs: {args.n_epochs}")
    print(f"   N steps: {args.n_steps}")
    
    # 3. 配置回调
    print(f"\n[3/5] 配置训练回调...")
    
    # 指标记录回调
    metrics_callback = MetricsCallback(log_dir=run_dir)
    
    # Checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix='ppo_lstm_stenosis',
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # 组合回调
    callback = CallbackList([
        metrics_callback,
        checkpoint_callback,
        eval_callback
    ])
    
    print(f"✅ 回调配置完成")
    print(f"   Checkpoint频率: 每 {args.save_freq} 步")
    print(f"   评估频率: 每 {args.eval_freq} 步")
    
    # 4. 开始训练
    print(f"\n[4/5] 开始训练...")
    print(f"   总步数: {args.total_timesteps}")
    print(f"   预计Episodes: ~{args.total_timesteps // args.max_episode_steps}")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠ 训练被中断")
    
    # 5. 保存最终模型
    print(f"\n[5/5] 保存最终模型...")
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print(f"✅ 模型已保存到: {final_model_path}.zip")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"\n输出目录: {run_dir}")
    print(f"  - 最终模型: {final_model_path}.zip")
    print(f"  - 最佳模型: {run_dir / 'best_model'}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - 训练曲线: {run_dir / 'training_curves.png'}")
    print(f"  - TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")


def main():
    parser = argparse.ArgumentParser(
        description="训练序列决策RL Agent进行狭窄检测"
    )
    
    # 数据集参数
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/mnt/sda1/luoyu/xzjc_data/dataset',
        help='数据集目录（包含.bmp和.xml）'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='/mnt/sda1/luoyu/xzjc_data/masks',
        help='Mask目录'
    )
    
    # 环境参数
    parser.add_argument(
        '--window_size',
        type=int,
        default=5,
        help='局部窗口大小（前后各N个点）'
    )
    parser.add_argument(
        '--max_episode_steps',
        type=int,
        default=500,
        help='每个Episode最大步数'
    )
    parser.add_argument(
        '--n_envs',
        type=int,
        default=1,
        help='并行环境数量（建议从1开始）'
    )
    
    # PPO参数
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='学习率'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=2048,
        help='每次更新的步数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=10,
        help='优化轮数'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='折扣因子'
    )
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=0.95,
        help='GAE lambda'
    )
    parser.add_argument(
        '--clip_range',
        type=float,
        default=0.2,
        help='PPO clip range'
    )
    parser.add_argument(
        '--ent_coef',
        type=float,
        default=0.01,
        help='熵系数（鼓励探索）'
    )
    
    # LSTM参数
    parser.add_argument(
        '--lstm_hidden_size',
        type=int,
        default=128,
        help='LSTM隐藏层大小'
    )
    
    # 训练参数
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=100000,
        help='总训练步数'
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=10000,
        help='Checkpoint保存频率'
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=5000,
        help='评估频率'
    )
    
    # 输出参数
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/mnt/sda1/luoyu/SAM-VMNet/rl_sequential_logs',
        help='日志目录'
    )
    
    args = parser.parse_args()
    
    # 运行训练
    train_agent(args)


if __name__ == '__main__':
    main()
