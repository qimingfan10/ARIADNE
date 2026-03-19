"""
对比RL自适应策略 vs 固定参数Statistical方法
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from rl_stenosis_env import StenosisDetectionEnv
from stenosis_detection.stenosis_detector import StenosisDetector


def evaluate_fixed_params(test_dataset, stenosis_threshold, min_avg_radius):
    """评估固定参数方法"""
    results = []
    
    print(f"\n评估固定参数方法 (threshold={stenosis_threshold}, min_radius={min_avg_radius})...")
    
    for sample in tqdm(test_dataset):
        try:
            detector = StenosisDetector(
                sample['image_path'],
                sample['mask_path'],
                resize_shape=(512, 512)
            )
            
            # 运行检测
            detector.extract_skeleton()
            detector.calculate_vessel_radius(max_radius=110, use_gpu=False)
            detector.find_segmentation_points()
            detector.filter_segmentation_points(min_distance=8)
            detector.detect_stenosis(
                stenosis_threshold=stenosis_threshold,
                min_avg_radius=min_avg_radius
            )
            
            detections = detector.all_stenosis_points if hasattr(detector, 'all_stenosis_points') else []
            if isinstance(detections, np.ndarray):
                detections = detections.tolist()
            
        except Exception as e:
            detections = []
        
        # 计算指标
        tp, fp, fn = match_detections(detections, sample['annotations'])
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results.append({
            'tp': tp, 'fp': fp, 'fn': fn,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1,
            'num_detections': len(detections),
            'num_gt': len(sample['annotations'])
        })
    
    return results


def evaluate_rl_adaptive(test_dataset, model_path):
    """评估RL自适应方法"""
    results = []
    actions_log = []
    
    print(f"\n评估RL自适应方法...")
    
    # 加载模型
    model = PPO.load(model_path)
    
    for sample in tqdm(test_dataset):
        try:
            # 创建环境
            env = StenosisDetectionEnv(
                dataset_path=os.path.dirname(sample['image_path']),
                annotations_csv=None,
                mask_path=os.path.dirname(sample['mask_path'])
            )
            
            # 手动设置当前样本
            env.image_path = sample['image_path']
            env.mask_path = sample['mask_path']
            env.ground_truth = sample['annotations']
            env.detector = StenosisDetector(
                sample['image_path'],
                sample['mask_path'],
                resize_shape=(512, 512)
            )
            
            # 获取状态
            state = env._extract_state()
            
            # Agent预测动作
            action, _ = model.predict(state, deterministic=True)
            
            # 记录动作
            actions_log.append({
                'stenosis_threshold': action[0],
                'min_avg_radius': action[1],
                'seg_distance': action[2]
            })
            
            # 执行动作
            _, reward, _, _, info = env.step(action)
            
            results.append({
                'tp': info['tp'],
                'fp': info['fp'],
                'fn': info['fn'],
                'sensitivity': info['sensitivity'],
                'precision': info['precision'],
                'f1_score': info['f1_score'],
                'num_detections': info['num_detections'],
                'num_gt': info['num_gt']
            })
            
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                'tp': 0, 'fp': 0, 'fn': len(sample['annotations']),
                'sensitivity': 0, 'precision': 0, 'f1_score': 0,
                'num_detections': 0, 'num_gt': len(sample['annotations'])
            })
            actions_log.append({
                'stenosis_threshold': 0.25,
                'min_avg_radius': 4.0,
                'seg_distance': 8.0
            })
    
    return results, actions_log


def match_detections(detections, ground_truth, threshold_mm=10.0):
    """匹配检测结果"""
    if len(detections) == 0:
        return 0, 0, len(ground_truth)
    
    if len(ground_truth) == 0:
        return 0, len(detections), 0
    
    pixel_spacing = 0.1953125
    threshold_pixels = threshold_mm / pixel_spacing
    
    gt_matched = [False] * len(ground_truth)
    tp = 0
    
    for det in detections:
        best_match = -1
        best_dist = float('inf')
        
        for i, gt in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            
            dist = np.sqrt((det[0] - gt[0])**2 + (det[1] - gt[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_match = i
        
        if best_match != -1 and best_dist <= threshold_pixels:
            gt_matched[best_match] = True
            tp += 1
    
    fp = len(detections) - tp
    fn = len(ground_truth) - tp
    
    return tp, fp, fn


def compare_and_visualize(fixed_results, rl_results, actions_log, save_dir):
    """对比并可视化结果"""
    
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    # 计算平均指标
    fixed_metrics = {
        'sensitivity': np.mean([r['sensitivity'] for r in fixed_results]),
        'precision': np.mean([r['precision'] for r in fixed_results]),
        'f1_score': np.mean([r['f1_score'] for r in fixed_results]),
        'tp': sum([r['tp'] for r in fixed_results]),
        'fp': sum([r['fp'] for r in fixed_results]),
        'fn': sum([r['fn'] for r in fixed_results])
    }
    
    rl_metrics = {
        'sensitivity': np.mean([r['sensitivity'] for r in rl_results]),
        'precision': np.mean([r['precision'] for r in rl_results]),
        'f1_score': np.mean([r['f1_score'] for r in rl_results]),
        'tp': sum([r['tp'] for r in rl_results]),
        'fp': sum([r['fp'] for r in rl_results]),
        'fn': sum([r['fn'] for r in rl_results])
    }
    
    # 打印对比
    print(f"\n固定参数方法:")
    print(f"  Sensitivity: {fixed_metrics['sensitivity']:.2%}")
    print(f"  Precision:   {fixed_metrics['precision']:.2%}")
    print(f"  F1-Score:    {fixed_metrics['f1_score']:.3f}")
    print(f"  TP/FP/FN:    {fixed_metrics['tp']}/{fixed_metrics['fp']}/{fixed_metrics['fn']}")
    
    print(f"\nRL自适应方法:")
    print(f"  Sensitivity: {rl_metrics['sensitivity']:.2%}")
    print(f"  Precision:   {rl_metrics['precision']:.2%}")
    print(f"  F1-Score:    {rl_metrics['f1_score']:.3f}")
    print(f"  TP/FP/FN:    {rl_metrics['tp']}/{rl_metrics['fp']}/{rl_metrics['fn']}")
    
    print(f"\n提升:")
    print(f"  Sensitivity: +{(rl_metrics['sensitivity'] - fixed_metrics['sensitivity']) * 100:.1f}%")
    print(f"  Precision:   +{(rl_metrics['precision'] - fixed_metrics['precision']) * 100:.1f}%")
    print(f"  F1-Score:    +{(rl_metrics['f1_score'] - fixed_metrics['f1_score']) * 100:.1f}%")
    print(f"  倍数提升:    {rl_metrics['f1_score'] / (fixed_metrics['f1_score'] + 1e-6):.1f}x")
    
    # 可视化
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 指标对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_names = ['Sensitivity', 'Precision', 'F1-Score']
    fixed_vals = [fixed_metrics['sensitivity'], fixed_metrics['precision'], fixed_metrics['f1_score']]
    rl_vals = [rl_metrics['sensitivity'], rl_metrics['precision'], rl_metrics['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, fixed_vals, width, label='Fixed Params', alpha=0.8)
    axes[0].bar(x + width/2, rl_vals, width, label='RL Adaptive', alpha=0.8)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. RL动作分布
    thresholds = [a['stenosis_threshold'] for a in actions_log]
    radii = [a['min_avg_radius'] for a in actions_log]
    
    axes[1].hist(thresholds, bins=20, alpha=0.7)
    axes[1].set_xlabel('Stenosis Threshold')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('RL Threshold Selection')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(radii, bins=20, alpha=0.7)
    axes[2].set_xlabel('Min Avg Radius')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('RL Radius Selection')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    print(f"\n✅ 对比图已保存: {save_dir}/comparison.png")
    
    # 3. 保存详细结果
    df_fixed = pd.DataFrame(fixed_results)
    df_rl = pd.DataFrame(rl_results)
    df_actions = pd.DataFrame(actions_log)
    
    df_fixed.to_csv(os.path.join(save_dir, 'fixed_results.csv'), index=False)
    df_rl.to_csv(os.path.join(save_dir, 'rl_results.csv'), index=False)
    df_actions.to_csv(os.path.join(save_dir, 'rl_actions.csv'), index=False)
    
    print(f"✅ 详细结果已保存: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='对比RL vs Fixed')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='RL模型路径')
    parser.add_argument('--test_csv', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/test_labels.csv',
                       help='测试集CSV')
    parser.add_argument('--dataset_path', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/dataset',
                       help='图像路径')
    parser.add_argument('--mask_path', type=str,
                       default='/mnt/sda1/luoyu/xzjc_data/masks',
                       help='Mask路径')
    parser.add_argument('--fixed_threshold', type=float, default=0.25,
                       help='固定参数：狭窄阈值')
    parser.add_argument('--fixed_radius', type=float, default=4.0,
                       help='固定参数：最小半径')
    parser.add_argument('--save_dir', type=str, default='./rl_comparison',
                       help='保存目录')
    parser.add_argument('--n_test', type=int, default=100,
                       help='测试样本数（-1为全部）')
    
    args = parser.parse_args()
    
    # 加载测试集
    env = StenosisDetectionEnv(
        dataset_path=args.dataset_path,
        annotations_csv=args.test_csv,
        mask_path=args.mask_path
    )
    test_dataset = env.dataset
    
    if args.n_test > 0:
        test_dataset = test_dataset[:args.n_test]
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 评估固定参数
    fixed_results = evaluate_fixed_params(
        test_dataset,
        args.fixed_threshold,
        args.fixed_radius
    )
    
    # 评估RL
    rl_results, actions_log = evaluate_rl_adaptive(
        test_dataset,
        args.model_path
    )
    
    # 对比和可视化
    compare_and_visualize(
        fixed_results,
        rl_results,
        actions_log,
        args.save_dir
    )


if __name__ == '__main__':
    main()
