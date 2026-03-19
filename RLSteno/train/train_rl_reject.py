#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练带拒绝机制的RL模型
目标：让Agent学会识别并拒绝假阳性，提升PPV
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.measure import label as sk_label
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from rl_reject_env import StenosisRejectEnv, create_mixed_training_data
from rl_sequential_stenosis_env import SequentialStenosisEnv


def get_main_skeleton_mask(skeleton_points, shape=(512, 512)):
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
        candidates.append({'idx': main_indices[i], 'score': score,
            'x': skeleton[main_indices[i]][1], 'y': skeleton[main_indices[i]][0]})
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


class RejectTrainingCallback(BaseCallback):
    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            env = self.training_env.envs[0]
            stats = env.get_stats()
            if self.verbose:
                print(f"\n[Step {self.n_calls}]")
                print(f"  TP={stats['tp']}, TN={stats['tn']}, FP={stats['fp']}, FN={stats['fn']}")
                print(f"  Precision={stats['precision']:.3f}, Recall={stats['recall']:.3f}, F1={stats['f1']:.3f}")
        return True


def main():
    print("="*60)
    print("训练带拒绝机制的RL模型")
    print("目标: 让Agent学会识别假阳性，提升PPV")
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
    
    # 2. 生成混合训练数据
    print("\n[2/4] 生成混合训练数据 (真阳性 + 假阳性)...")
    train_data = create_mixed_training_data(
        base_env, 
        base_env.cache, 
        clean_indices, 
        detect_stenosis,
        fp_ratio=0.4  # 假阳性占40%
    )
    
    if len(train_data) == 0:
        print("❌ 没有生成训练数据")
        return
    
    # 3. 创建环境
    print("\n[3/4] 创建RL环境...")
    
    def make_env():
        return StenosisRejectEnv(train_data, window_size=30, max_steps=50)
    
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
        ent_coef=0.02,  # 稍高熵值，鼓励探索拒绝动作
        verbose=1,
        device='cpu',
        tensorboard_log="./tensorboard_logs/rl_reject"
    )
    
    callback = RejectTrainingCallback(eval_freq=10000)
    
    total_timesteps = 150000
    print(f"  总训练步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    model_path = "rl_reject_model"
    model.save(model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 5. 评估
    print("\n" + "="*60)
    print("评估带拒绝机制的模型")
    print("="*60)
    
    evaluate_reject_model(model, train_data, base_env, clean_indices)


def evaluate_reject_model(model, test_data, base_env, clean_indices):
    """评估带拒绝机制的模型"""
    from scipy.ndimage import gaussian_filter1d
    
    env = StenosisRejectEnv(test_data)
    
    # 统计
    results = {
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'confirmed_dists': [],
        'rejected_tp': 0,
        'rejected_fp': 0
    }
    
    for i in range(len(test_data)):
        sample = test_data[i]
        
        # 设置环境
        env.current_sample = sample
        env.skeleton = sample['skeleton']
        env.radii = sample['radii']
        env.skel_len = len(env.radii)
        env.curr_pos = sample['start_idx']
        env.target_idx = sample.get('target_idx', None)
        env.is_false_positive = sample.get('is_fp', False)
        env.steps = 0
        
        if env.target_idx is not None:
            env.prev_pixel_dist = env._calc_pixel_dist(env.curr_pos, env.target_idx)
        else:
            env.prev_pixel_dist = float('inf')
        
        obs = env._get_obs()
        done = False
        final_action = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            if term:
                final_action = action
        
        # 统计结果
        is_tp = not sample.get('is_fp', False)  # 真阳性样本
        
        if final_action == 2:  # Confirm
            if is_tp:
                results['tp'] += 1
                if env.target_idx is not None:
                    dist = env._calc_pixel_dist(env.curr_pos, env.target_idx)
                    results['confirmed_dists'].append(dist)
            else:
                results['fp'] += 1
        elif final_action == 3:  # Reject
            if is_tp:
                results['fn'] += 1
                results['rejected_tp'] += 1
            else:
                results['tn'] += 1
                results['rejected_fp'] += 1
    
    # 计算指标
    precision = results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0
    recall = results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n样本数: {len(test_data)}")
    print(f"  真阳性样本: {sum(1 for d in test_data if not d.get('is_fp', False))}")
    print(f"  假阳性样本: {sum(1 for d in test_data if d.get('is_fp', False))}")
    
    print(f"\n混淆矩阵:")
    print(f"  TP (正确确认): {results['tp']}")
    print(f"  TN (正确拒绝): {results['tn']}")
    print(f"  FP (错误确认): {results['fp']}")
    print(f"  FN (错误拒绝): {results['fn']}")
    
    print(f"\n指标:")
    print(f"  Precision (PPV): {precision:.3f}")
    print(f"  Recall (TPR):    {recall:.3f}")
    print(f"  F1 Score:        {f1:.3f}")
    
    if results['confirmed_dists']:
        print(f"\n确认样本的定位精度:")
        print(f"  平均距离: {np.mean(results['confirmed_dists']):.1f}px")
        print(f"  ≤20px: {sum(1 for d in results['confirmed_dists'] if d <= 20)}/{len(results['confirmed_dists'])}")
        print(f"  ≤40px: {sum(1 for d in results['confirmed_dists'] if d <= 40)}/{len(results['confirmed_dists'])}")
    
    print(f"\n拒绝统计:")
    print(f"  正确拒绝假阳性: {results['rejected_fp']}")
    print(f"  错误拒绝真阳性: {results['rejected_tp']}")
    
    return results


if __name__ == "__main__":
    main()
