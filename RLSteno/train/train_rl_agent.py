"""
训练RL Agent进行自适应狭窄检测
"""
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from rl_stenosis_env import StenosisDetectionEnv


class MetricsCallback(CheckpointCallback):
    """记录训练指标的回调"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = []
        self.f1_scores = []
        self.sensitivities = []
        self.precisions = []
    
    def _on_step(self):
        # 记录metrics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'f1_score' in info:
                    self.f1_scores.append(info['f1_score'])
                    self.sensitivities.append(info.get('sensitivity', 0))
                    self.precisions.append(info.get('precision', 0))
        
        return super()._on_step()


def train_rl_agent(args):
    """训练RL Agent"""
    
    print("=" * 60)
    print("训练RL Agent进行自适应狭窄检测")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. 创建环境
    print(f"\n[1/5] 创建训练环境...")
    
    def make_env():
        env = StenosisDetectionEnv(
            dataset_path=args.dataset_path,
            annotations_csv=args.train_csv,
            mask_path=args.mask_path,
            mode='train'
        )
        env = Monitor(env)
        return env
    
    # 并行环境
    vec_env = make_vec_env(make_env, n_envs=args.n_envs)
    print(f"✅ 创建了 {args.n_envs} 个并行环境")
    
    # 评估环境
    eval_env = Monitor(StenosisDetectionEnv(
        dataset_path=args.dataset_path,
        annotations_csv=args.test_csv if args.test_csv else args.train_csv,
        mask_path=args.mask_path,
        mode='test'
    ))
    
    # 2. 配置PPO模型
    print(f"\n[2/5] 配置PPO模型...")
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )
    
    print(f"✅ PPO配置:")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   N epochs: {args.n_epochs}")
    
    # 3. 配置回调
    print(f"\n[3/5] 配置回调函数...")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = MetricsCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=args.checkpoint_dir,
        name_prefix='rl_stenosis',
        verbose=1
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # 4. 训练
    print(f"\n[4/5] 开始训练...")
    print(f"   总步数: {args.total_timesteps}")
    print(f"   估计时间: {args.total_timesteps / args.n_envs / 60:.1f}分钟")
    print(f"   TensorBoard: tensorboard --logdir {args.log_dir}")
    print()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    # 5. 保存最终模型
    print(f"\n[5/5] 保存最终模型...")
    final_model_path = os.path.join(args.checkpoint_dir, 'rl_stenosis_final')
    model.save(final_model_path)
    print(f"✅ 模型已保存: {final_model_path}")
    
    # 可视化训练指标
    if len(checkpoint_callback.f1_scores) > 0:
        plot_training_metrics(checkpoint_callback, args.log_dir)
    
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)
    
    return model


def plot_training_metrics(callback, save_dir):
    """可视化训练指标"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # F1-Score
    if len(callback.f1_scores) > 0:
        axes[0].plot(callback.f1_scores, alpha=0.3)
        axes[0].plot(np.convolve(callback.f1_scores, np.ones(100)/100, mode='valid'))
        axes[0].set_title('F1-Score')
        axes[0].set_xlabel('Episode')
        axes[0].grid(True)
    
    # Sensitivity
    if len(callback.sensitivities) > 0:
        axes[1].plot(callback.sensitivities, alpha=0.3)
        axes[1].plot(np.convolve(callback.sensitivities, np.ones(100)/100, mode='valid'))
        axes[1].set_title('Sensitivity')
        axes[1].set_xlabel('Episode')
        axes[1].grid(True)
    
    # Precision
    if len(callback.precisions) > 0:
        axes[2].plot(callback.precisions, alpha=0.3)
        axes[2].plot(np.convolve(callback.precisions, np.ones(100)/100, mode='valid'))
        axes[2].set_title('Precision')
        axes[2].set_xlabel('Episode')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    print(f"✅ 训练曲线已保存: {save_dir}/training_metrics.png")


def main():
    parser = argparse.ArgumentParser(description='训练RL Agent')
    
    # 数据集参数
    parser.add_argument('--dataset_path', type=str, 
                       default='/mnt/sda1/luoyu/xzjc_data/dataset',
                       help='图像数据集路径')
    parser.add_argument('--train_csv', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/train_labels.csv',
                       help='训练集CSV')
    parser.add_argument('--test_csv', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/test_labels.csv',
                       help='测试集CSV')
    parser.add_argument('--mask_path', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/masks',
                       help='Mask路径')
    
    # 训练参数
    parser.add_argument('--n_envs', type=int, default=4,
                       help='并行环境数')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                       help='总训练步数')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='每次更新的步数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='每次更新的epoch数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2,
                       help='PPO clip范围')
    
    # 评估和保存
    parser.add_argument('--eval_freq', type=int, default=5000,
                       help='评估频率')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                       help='评估episode数')
    parser.add_argument('--save_freq', type=int, default=10000,
                       help='保存频率')
    parser.add_argument('--log_dir', type=str, default='./rl_logs',
                       help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./rl_checkpoints',
                       help='检查点目录')
    
    args = parser.parse_args()
    
    # 训练
    model = train_rl_agent(args)


if __name__ == '__main__':
    main()
