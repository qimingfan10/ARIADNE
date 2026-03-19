"""
使用Reward Network和RL微调Sa2VA模型
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 添加Sa2VA路径
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.sa2va_finetune_env import Sa2VAFinetuneEnv
from models.reward_network import LightweightRewardNetwork


class TensorBoardCallback(BaseCallback):
    """
    自定义TensorBoard回调，记录自定义指标
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_gt_dices = []
        self.episode_reward_net_scores = []
    
    def _on_step(self) -> bool:
        # 从info中提取指标
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Episode结束
                    self.logger.record('rollout/ep_rew_mean', info['episode']['r'])
                    self.logger.record('rollout/ep_len_mean', info['episode']['l'])
                
                # 记录自定义指标
                if 'reward_net_score' in info:
                    self.logger.record('custom/reward_net_score', info['reward_net_score'])
                if 'gt_dice' in info:
                    self.logger.record('custom/gt_dice', info['gt_dice'])
                if 'best_reward' in info:
                    self.logger.record('custom/best_reward', info['best_reward'])
        
        return True


def load_reward_network(reward_net_path, device='cuda'):
    """加载训练好的Reward Network"""
    print(f"\n加载Reward Network: {reward_net_path}")
    
    reward_net = LightweightRewardNetwork(input_channels=4).to(device)
    
    checkpoint = torch.load(reward_net_path, map_location=device)
    reward_net.load_state_dict(checkpoint['model_state_dict'])
    reward_net.eval()
    
    print(f"✅ Reward Network加载成功")
    return reward_net


def load_sa2va_model(model_path, device='cuda'):
    """加载Sa2VA模型"""
    print(f"\n加载Sa2VA模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    print(f"✅ Sa2VA模型加载成功")
    
    return model, tokenizer


def load_dataset(data_root, split='train', max_samples=None):
    """加载数据集"""
    from PIL import Image, ImageDraw
    import cv2
    
    # 加载annotations
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 限制样本数
    if max_samples is not None:
        annotations = annotations[:max_samples]
    
    # 加载数据
    dataset = []
    images_dir = os.path.join(data_root, 'images')
    
    for ann in tqdm(annotations, desc=f"加载{split}数据"):
        try:
            # 加载图像
            image_path = os.path.join(images_dir, ann['image'])
            image = Image.open(image_path).convert('RGB')
            
            # 从坐标点生成mask
            width, height = image.size
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            
            # 绘制多边形mask
            if 'mask' in ann and len(ann['mask']) > 0:
                # mask是坐标点列表
                polygons = ann['mask']
                for polygon in polygons:
                    if len(polygon) >= 6:  # 至少3个点
                        # 转换为(x,y)坐标对
                        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                        draw.polygon(points, fill=255)
            
            # 转换为numpy数组
            mask = np.array(mask_img)
            
            dataset.append({
                'image': image,
                'mask': mask,
                'image_path': ann['image']
            })
            
        except Exception as e:
            print(f"加载失败 {ann.get('image', 'unknown')}: {e}")
            continue
    
    print(f"✅ 加载{len(dataset)}个样本")
    return dataset


def make_env(sa2va_model, tokenizer, reward_network, dataset, device='cuda'):
    """创建环境工厂函数"""
    def _init():
        env = Sa2VAFinetuneEnv(
            sa2va_model=sa2va_model,
            tokenizer=tokenizer,
            reward_network=reward_network,
            dataset=dataset,
            device=device,
            max_steps=1  # 每个episode只选择一次prompt
        )
        env = Monitor(env)
        return env
    return _init


def train_sa2va_with_rl(args):
    """使用RL微调Sa2VA"""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'sa2va_rl_finetune_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Reward Network
    reward_network = load_reward_network(args.reward_net_path, device)
    
    # 加载Sa2VA模型
    sa2va_model, tokenizer = load_sa2va_model(args.model_path, device)
    
    # 加载数据集
    dataset = load_dataset(args.data_root, split='train', max_samples=args.max_samples)
    
    print("\n" + "=" * 80)
    print("创建RL环境...")
    print("=" * 80)
    
    # 创建向量化环境
    env = DummyVecEnv([
        make_env(sa2va_model, tokenizer, reward_network, dataset, device)
        for _ in range(args.n_envs)
    ])
    
    print(f"✅ 创建了{args.n_envs}个并行环境")
    
    # 配置PPO
    print("\n" + "=" * 80)
    print("配置PPO算法...")
    print("=" * 80)
    
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])]
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(output_dir, 'logs'),
        verbose=1
    )
    
    print(f"✅ PPO配置完成")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - N Steps: {args.n_steps}")
    print(f"  - N Epochs: {args.n_epochs}")
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(output_dir, 'checkpoints'),
        name_prefix='sa2va_rl'
    )
    
    tensorboard_callback = TensorBoardCallback()
    
    callbacks = [checkpoint_callback, tensorboard_callback]
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始RL微调...")
    print("=" * 80)
    print(f"总训练步数: {args.total_timesteps}")
    print(f"输出目录: {output_dir}")
    print("=" * 80 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model')
        model.save(final_model_path)
        print(f"\n✅ 训练完成！模型保存至: {final_model_path}")
        
        # 保存训练信息
        info = {
            'timestamp': timestamp,
            'args': vars(args),
            'reward_net_path': args.reward_net_path,
            'model_path': args.model_path,
            'total_timesteps': args.total_timesteps,
            'output_dir': output_dir
        }
        
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        return output_dir
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description='使用Reward Network微调Sa2VA')
    
    # 路径参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Sa2VA模型路径')
    parser.add_argument('--reward_net_path', type=str, required=True,
                        help='训练好的Reward Network路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--max_samples', type=int, default=50,
                        help='最大训练样本数')
    parser.add_argument('--total_timesteps', type=int, default=5000,
                        help='总训练步数')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='并行环境数量')
    parser.add_argument('--n_steps', type=int, default=128,
                        help='每次更新的步数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='每次更新的epoch数')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='保存频率')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("实验三步骤2：使用Reward Network微调Sa2VA")
    print("=" * 80)
    print(f"Sa2VA模型: {args.model_path}")
    print(f"Reward Network: {args.reward_net_path}")
    print(f"数据集: {args.data_root}")
    print(f"训练样本: {args.max_samples}")
    print(f"总步数: {args.total_timesteps}")
    print("=" * 80 + "\n")
    
    output_dir = train_sa2va_with_rl(args)
    
    if output_dir:
        print("\n" + "=" * 80)
        print("训练成功完成！")
        print("=" * 80)
        print(f"模型保存位置: {output_dir}")
        print(f"TensorBoard: tensorboard --logdir {output_dir}/logs")
        print("=" * 80)
    else:
        print("\n训练失败，请检查错误信息")
        sys.exit(1)


if __name__ == '__main__':
    main()
