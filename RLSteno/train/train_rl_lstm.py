#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM版本的RL精定位训练
使用RecurrentPPO替代PPO，引入记忆机制
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from skimage.measure import label as sk_label
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from rl_localization_env import StenosisLocalizationEnv, create_training_data
from rl_sequential_stenosis_env import SequentialStenosisEnv


def get_main_skeleton_mask(skeleton_points, shape=(512, 512)):
    """获取主连通区域掩码"""
    skeleton_img = np.zeros(shape, dtype=np.uint8)
    for pt in skeleton_points:
        r, c = int(pt[0]), int(pt[1])
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            skeleton_img[r, c] = 255
    labeled, n_labels = sk_label(skeleton_img, return_num=True, connectivity=2)
    if n_labels == 0:
        return np.ones(len(skeleton_points), dtype=bool)
    region_sizes = np.bincount(labeled.ravel())
    region_sizes[0] = 0
    main_label = np.argmax(region_sizes)
    main_mask = np.zeros(len(skeleton_points), dtype=bool)
    for i, pt in enumerate(skeleton_points):
        r, c = int(pt[0]), int(pt[1])
        if labeled[r, c] == main_label:
            main_mask[i] = True
    return main_mask


def detect_stenosis(skeleton, radii, main_mask, top_n=3, min_dist=40):
    """几何检测算法（粗定位）"""
    main_indices = np.where(main_mask)[0]
    if len(main_indices) < 30:
        return []
    
    main_radii = np.array([radii[i] for i in main_indices])
    n = len(main_radii)
    margin = max(5, int(n * 0.05))
    
    smooth = gaussian_filter1d(main_radii.astype(float), sigma=2)
    grad = np.gradient(smooth)
    curv = np.gradient(grad)
    
    candidates = []
    window = 6
    
    for i in range(margin + window, n - margin - window):
        r = main_radii[i]
        local = main_radii[max(0, i-window):min(n, i+window+1)]
        local_smooth = smooth[max(0, i-window):min(n, i+window+1)]
        
        is_local_min = (r == np.min(local)) or (smooth[i] <= np.min(local_smooth) * 1.02)
        if not is_local_min:
            continue
        
        depth = np.max(local) - r
        curvature_score = max(0, curv[i])
        global_rank = np.sum(main_radii < r) / n
        score = depth * (1 + curvature_score) * (1 - global_rank)
        
        candidates.append({
            'idx': main_indices[i],
            'score': score,
            'x': skeleton[main_indices[i]][1],
            'y': skeleton[main_indices[i]][0]
        })
    
    if not candidates:
        return []
    
    candidates.sort(key=lambda x: -x['score'])
    
    keep = []
    for c in candidates:
        is_dup = any(np.sqrt((c['x']-k['x'])**2 + (c['y']-k['y'])**2) < min_dist for k in keep)
        if not is_dup:
            keep.append(c)
            if len(keep) >= top_n:
                break
    
    return [c['idx'] for c in keep]


class LSTMTrainingCallback(BaseCallback):
    """LSTM训练过程回调"""
    
    def __init__(self, eval_freq=2000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_success_rate = 0
        self.rewards = []
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            env = self.training_env.envs[0]
            success_rate = env.get_success_rate()
            
            if self.verbose:
                print(f"\n[Step {self.n_calls}] Success Rate: {success_rate:.3f}")
            
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                if self.verbose:
                    print(f"  ★ New best: {success_rate:.3f}")
        
        return True


def main():
    print("="*60)
    print("LSTM RL精定位训练")
    print("目标: 利用记忆机制进一步提升定位精度")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    base_env = SequentialStenosisEnv(
        dataset_dir="/mnt/sda1/luoyu/xzjc_data/dataset",
        mask_dir="/mnt/sda1/luoyu/xzjc_data/masks",
        cache_file="skeleton_cache_v2.pkl"
    )
    
    clean_indices = np.load('clean_vessel_indices.npy')
    print(f"  干净样本数: {len(clean_indices)}")
    
    # 2. 生成训练数据
    print("\n[2/4] 生成训练数据...")
    train_data = create_training_data(
        base_env, 
        base_env.cache, 
        clean_indices, 
        detect_stenosis,
        margin_threshold=100
    )
    
    if len(train_data) == 0:
        print("❌ 没有生成训练数据")
        return
    
    print(f"  训练样本数: {len(train_data)}")
    
    # 3. 创建环境
    print("\n[3/4] 创建RL环境...")
    
    def make_env():
        return StenosisLocalizationEnv(train_data, window_size=30, max_steps=50)
    
    env = DummyVecEnv([make_env])
    
    # 4. 配置LSTM模型
    print("\n[4/4] 配置LSTM模型...")
    
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[64, 64],  # Actor网络
            vf=[64, 64]   # Critic网络
        ),
        lstm_hidden_size=128,    # LSTM隐藏层大小
        n_lstm_layers=1,         # LSTM层数
        shared_lstm=False,       # Actor/Critic不共享LSTM
        enable_critic_lstm=True  # Critic也使用LSTM
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,        # LSTM需要较短的序列
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cpu',  # LSTM在CPU上更稳定
        tensorboard_log="./tensorboard_logs/rl_lstm"
    )
    
    # 训练
    callback = LSTMTrainingCallback(eval_freq=5000)
    
    total_timesteps = 100000
    print(f"\n开始训练 (总步数: {total_timesteps})")
    print("-"*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    model_path = "rl_lstm_model"
    model.save(model_path)
    print(f"\n✅ LSTM模型已保存: {model_path}")
    
    # 5. 评估并对比MLP
    print("\n" + "="*60)
    print("评估LSTM vs MLP对比")
    print("="*60)
    
    evaluate_and_compare(model, train_data)


def evaluate_and_compare(lstm_model, test_data, n_episodes=None):
    """评估LSTM模型并与MLP对比"""
    from stable_baselines3 import PPO
    
    if n_episodes is None:
        n_episodes = len(test_data)
    
    env = StenosisLocalizationEnv(test_data)
    
    # 加载MLP模型
    try:
        mlp_model = PPO.load("rl_localization_model")
        has_mlp = True
    except:
        has_mlp = False
        print("⚠ 未找到MLP模型，只评估LSTM")
    
    results = {'lstm': [], 'mlp': [], 'geo': []}
    
    for i in range(min(n_episodes, len(test_data))):
        sample = test_data[i]
        
        # 设置环境
        env.current_sample = sample
        env.skeleton = sample['skeleton']
        env.radii = sample['radii']
        env.skel_len = len(env.radii)
        env.curr_pos = sample['start_idx']
        env.target_idx = sample['target_idx']
        env.steps = 0
        env.prev_pixel_dist = env._calc_pixel_dist(env.curr_pos, env.target_idx)
        
        geo_dist = env.prev_pixel_dist
        results['geo'].append(geo_dist)
        
        # LSTM评估
        obs = env._get_obs()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        done = False
        env.curr_pos = sample['start_idx']
        env.steps = 0
        env.prev_pixel_dist = geo_dist
        
        while not done:
            action, lstm_states = lstm_model.predict(
                obs.reshape(1, -1), 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
            obs, reward, term, trunc, info = env.step(action[0])
            done = term or trunc
        
        results['lstm'].append(info['pixel_dist'])
        
        # MLP评估
        if has_mlp:
            env.curr_pos = sample['start_idx']
            env.steps = 0
            env.prev_pixel_dist = geo_dist
            obs = env._get_obs()
            
            done = False
            while not done:
                action, _ = mlp_model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
            
            results['mlp'].append(info['pixel_dist'])
    
    # 统计结果
    margins = [20, 30, 40, 50]
    
    print(f"\n样本数: {len(results['geo'])}")
    print(f"\n{'方法':<15} {'平均距离':<12} {'Margin=20':<10} {'Margin=30':<10} {'Margin=40':<10}")
    print("-"*60)
    
    for name, dists in [('几何算法', results['geo']), ('MLP', results['mlp']), ('LSTM', results['lstm'])]:
        if len(dists) == 0:
            continue
        recalls = [np.mean([d <= m for d in dists]) * 100 for m in margins]
        print(f"{name:<15} {np.mean(dists):.1f}px{'':<5} {recalls[0]:.1f}%{'':<5} {recalls[1]:.1f}%{'':<5} {recalls[2]:.1f}%")
    
    print("-"*60)
    
    # 改善分析
    if has_mlp:
        lstm_better = sum(1 for l, m in zip(results['lstm'], results['mlp']) if l < m)
        print(f"\nLSTM优于MLP的样本比例: {lstm_better}/{len(results['lstm'])} ({lstm_better/len(results['lstm'])*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    main()
