#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练RL精定位模型
目标：将几何算法的粗定位精确到margin=20
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.measure import label as sk_label
from stable_baselines3 import PPO
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


class TrainingCallback(BaseCallback):
    """训练过程回调"""
    
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_success_rate = 0
        
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
    print("RL精定位训练")
    print("目标: 将几何算法的粗定位(margin=75)精确到margin=20")
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
        print("❌ 没有生成训练数据，请检查数据")
        return
    
    # 统计初始距离分布
    init_dists = [d['init_dist'] for d in train_data]
    print(f"  初始距离分布:")
    print(f"    Mean: {np.mean(init_dists):.1f}px")
    print(f"    Std:  {np.std(init_dists):.1f}px")
    print(f"    Min:  {np.min(init_dists):.1f}px")
    print(f"    Max:  {np.max(init_dists):.1f}px")
    
    # 3. 创建环境
    print("\n[3/4] 创建RL环境...")
    
    def make_env():
        return StenosisLocalizationEnv(train_data, window_size=30, max_steps=50)
    
    env = DummyVecEnv([make_env])
    
    # 4. 训练
    print("\n[4/4] 开始训练...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard_logs/rl_localization"
    )
    
    callback = TrainingCallback(eval_freq=5000)
    
    total_timesteps = 100000
    print(f"  总训练步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    model_path = "rl_localization_model"
    model.save(model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 5. 评估
    print("\n" + "="*60)
    print("评估训练结果")
    print("="*60)
    
    evaluate_model(model, train_data)


def evaluate_model(model, test_data, n_episodes=100):
    """评估模型性能"""
    
    env = StenosisLocalizationEnv(test_data)
    
    results = {
        'margin_20': 0,
        'margin_30': 0,
        'margin_40': 0,
        'margin_50': 0,
        'total': 0
    }
    
    init_dists = []
    final_dists = []
    
    for _ in range(min(n_episodes, len(test_data))):
        obs, info = env.reset()
        init_dist = env.prev_pixel_dist
        init_dists.append(init_dist)
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        
        final_dist = info['pixel_dist']
        final_dists.append(final_dist)
        
        results['total'] += 1
        if final_dist <= 20:
            results['margin_20'] += 1
        if final_dist <= 30:
            results['margin_30'] += 1
        if final_dist <= 40:
            results['margin_40'] += 1
        if final_dist <= 50:
            results['margin_50'] += 1
    
    print(f"\n评估结果 (n={results['total']}):")
    print(f"  初始距离: {np.mean(init_dists):.1f} ± {np.std(init_dists):.1f} px")
    print(f"  最终距离: {np.mean(final_dists):.1f} ± {np.std(final_dists):.1f} px")
    print(f"  距离改善: {np.mean(init_dists) - np.mean(final_dists):.1f} px")
    print()
    print(f"  Margin=20: {results['margin_20']/results['total']*100:.1f}%")
    print(f"  Margin=30: {results['margin_30']/results['total']*100:.1f}%")
    print(f"  Margin=40: {results['margin_40']/results['total']*100:.1f}%")
    print(f"  Margin=50: {results['margin_50']/results['total']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    main()
