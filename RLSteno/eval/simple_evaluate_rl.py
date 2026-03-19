#!/usr/bin/env python3
"""简化的RL模型评估脚本 - 直接使用环境"""
import numpy as np
from stable_baselines3 import PPO
from rl_stenosis_env import StenosisDetectionEnv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model_path, n_test=100):
    """评估RL模型"""
    print("=" * 60)
    print(f"评估RL模型: {model_path}")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    model = PPO.load(model_path)
    
    # 创建环境
    print("创建环境...")
    env = StenosisDetectionEnv()
    
    # 统计
    all_actions = []
    all_metrics = []
    
    print(f"\n测试 {n_test} 个样本...")
    
    for i in tqdm(range(n_test)):
        # 重置环境
        state, _ = env.reset()
        
        # Agent预测动作
        action, _ = model.predict(state, deterministic=True)
        all_actions.append(action)
        
        # 执行动作
        _, reward, _, _, info = env.step(action)
        
        all_metrics.append({
            'tp': info['tp'],
            'fp': info['fp'],
            'fn': info['fn'],
            'f1': info['f1_score'],
            'threshold': action[0],
            'min_radius': action[1],
            'seg_distance': action[2]
        })
    
    # 统计结果
    actions = np.array(all_actions)
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    print(f"\n检测指标:")
    print(f"  TP: {total_tp}")
    print(f"  FP: {total_fp}")
    print(f"  FN: {total_fn}")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  平均F1: {avg_f1:.3f}")
    
    print(f"\n策略统计:")
    print(f"  Threshold: {actions[:,0].mean():.3f} ± {actions[:,0].std():.3f}")
    print(f"             范围 [{actions[:,0].min():.3f}, {actions[:,0].max():.3f}]")
    print(f"  MinRadius: {actions[:,1].mean():.2f} ± {actions[:,1].std():.2f}")
    print(f"             范围 [{actions[:,1].min():.2f}, {actions[:,1].max():.2f}]")
    print(f"  SegDist:   {actions[:,2].mean():.1f} ± {actions[:,2].std():.1f}")
    print(f"             范围 [{actions[:,2].min():.1f}, {actions[:,2].max():.1f}]")
    
    # 判断自适应性
    threshold_std = actions[:,0].std()
    if threshold_std < 0.01:
        print(f"\n⚠️  警告: Threshold标准差{threshold_std:.4f}太小")
        print(f"   Agent可能没有学会自适应策略")
    else:
        print(f"\n✅ Threshold有{threshold_std:.3f}的变化，显示一定自适应能力")
    
    # 展示几个样本的策略
    print(f"\n策略示例（前10个样本）:")
    print(f"{'样本':<6} {'Threshold':<11} {'MinRadius':<11} {'SegDist':<10} {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 60)
    for i in range(min(10, len(all_metrics))):
        m = all_metrics[i]
        print(f"{i+1:<6} {m['threshold']:<11.3f} {m['min_radius']:<11.2f} {m['seg_distance']:<10.1f} "
              f"{m['tp']:<4} {m['fp']:<4} {m['fn']:<4}")
    
    print("=" * 60)
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'sensitivity': sensitivity,
        'precision': precision,
        'avg_f1': avg_f1,
        'actions': actions,
        'metrics': all_metrics
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    else:
        # 默认评估30k步模型
        model_path = './rl_checkpoints_full/rl_stenosis_30000_steps.zip'
        n_test = 100
    
    results = evaluate_model(model_path, n_test)
