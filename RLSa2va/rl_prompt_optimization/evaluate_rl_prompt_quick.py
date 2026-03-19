"""
快速评估训练好的Prompt优化策略（仅评估少量样本）
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/Sa2VA')
from rl_prompt_optimization.env.prompt_env import PromptOptimizationEnv


def load_dataset(data_root, max_samples=50):
    """加载数据集（限制数量）"""
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        all_data = json.load(f)
    
    # 简单划分：后20%作为验证集
    n_train = int(len(all_data) * 0.8)
    val_data = all_data[n_train:]
    
    # 只取前max_samples张
    data = val_data[:max_samples]
    
    # 添加完整路径
    for item in data:
        item['image_path'] = os.path.join(data_root, 'images', item['image'])
    
    print(f"加载{len(data)}张图像进行快速评估")
    return data


def evaluate(args):
    """评估主函数"""
    print("=" * 80)
    print("Sa2VA Prompt优化策略快速评估")
    print(f"样本数: {args.max_samples}")
    print(f"GPU: {args.gpu}")
    print("=" * 80)
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_quick_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Sa2VA模型
    print("\n加载Sa2VA模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
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
    
    # 加载数据集（限制数量）
    print("\n加载测试数据集...")
    test_dataset = load_dataset(args.data_root, max_samples=args.max_samples)
    
    # 加载RL策略
    print(f"\n加载RL策略: {args.rl_model_path}")
    rl_policy = PPO.load(args.rl_model_path)
    print("✅ RL策略加载成功")
    
    # 创建环境
    env = PromptOptimizationEnv(model, tokenizer, test_dataset, max_steps=args.max_steps)
    
    # 评估
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    all_results = []
    prompt_stats = defaultdict(lambda: {'count': 0, 'dice_sum': 0, 'recall_sum': 0})
    
    for i in range(len(test_dataset)):
        print(f"\n[{i+1}/{len(test_dataset)}] 评估样本: {test_dataset[i]['image']}")
        
        obs, info = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_prompts = []
        
        try:
            for step in range(args.max_steps):
                # 使用RL策略选择动作
                action, _states = rl_policy.predict(obs, deterministic=True)
                action = int(action)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_rewards.append(reward)
                episode_actions.append(action)
                episode_prompts.append(info['prompt'])
                
                print(f"  步骤{step+1}: Prompt#{action}, "
                      f"Dice={info['dice']:.4f}, "
                      f"Recall={info['recall']:.4f}, "
                      f"Reward={reward:.2f}")
                
                # 统计prompt使用
                prompt_stats[action]['count'] += 1
                prompt_stats[action]['dice_sum'] += info['dice']
                prompt_stats[action]['recall_sum'] += info['recall']
                
                if terminated or truncated:
                    break
            
            # 记录结果
            result = {
                'sample_id': i + 1,
                'image': test_dataset[i]['image'],
                'best_dice': info['best_dice'],
                'best_recall': info['best_recall'],
                'total_reward': sum(episode_rewards),
                'actions': episode_actions,
                'prompts': episode_prompts,
                'final_dice': info['dice'],
                'final_recall': info['recall'],
                'final_precision': info['precision']
            }
            all_results.append(result)
            
            print(f"  最佳结果: Dice={info['best_dice']:.4f}, Recall={info['best_recall']:.4f}")
            
        except Exception as e:
            print(f"  ⚠️ 评估失败: {e}")
            continue
    
    # 统计总体结果
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    
    if len(all_results) > 0:
        avg_dice = np.mean([r['best_dice'] for r in all_results])
        avg_recall = np.mean([r['best_recall'] for r in all_results])
        avg_precision = np.mean([r['final_precision'] for r in all_results])
        
        print(f"\n平均指标 (基于{len(all_results)}个成功样本):")
        print(f"  Dice Score:  {avg_dice:.4f}")
        print(f"  Recall:      {avg_recall:.4f}")
        print(f"  Precision:   {avg_precision:.4f}")
        
        # Prompt统计
        print(f"\nPrompt使用统计:")
        print(f"{'Prompt ID':<12} {'使用次数':<12} {'平均Dice':<12} {'平均Recall':<12}")
        print("-" * 50)
        for prompt_id, stats in sorted(prompt_stats.items()):
            avg_dice_prompt = stats['dice_sum'] / stats['count']
            avg_recall_prompt = stats['recall_sum'] / stats['count']
            print(f"{prompt_id:<12} {stats['count']:<12} {avg_dice_prompt:<12.4f} {avg_recall_prompt:<12.4f}")
        
        # 保存结果
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'avg_dice': float(avg_dice),
                    'avg_recall': float(avg_recall),
                    'avg_precision': float(avg_precision),
                    'num_samples': len(all_results),
                    'total_samples_attempted': len(test_dataset)
                },
                'prompt_stats': {
                    k: {
                        'count': v['count'],
                        'avg_dice': v['dice_sum'] / v['count'],
                        'avg_recall': v['recall_sum'] / v['count']
                    }
                    for k, v in prompt_stats.items()
                },
                'per_sample_results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 评估结果保存至: {results_path}")
        
        # 绘制结果图
        plot_results(all_results, prompt_stats, output_dir)
    else:
        print("\n❌ 没有成功评估的样本")
    
    return output_dir


def plot_results(all_results, prompt_stats, output_dir):
    """绘制评估结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Dice分布
    dices = [r['best_dice'] for r in all_results]
    axes[0, 0].hist(dices, bins=20, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(dices), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(dices):.4f}')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].legend()
    
    # 2. Recall分布
    recalls = [r['best_recall'] for r in all_results]
    axes[0, 1].hist(recalls, bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(np.mean(recalls), color='red', linestyle='--',
                       label=f'Mean: {np.mean(recalls):.4f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Recall Distribution')
    axes[0, 1].legend()
    
    # 3. Prompt使用频率
    prompt_ids = sorted(prompt_stats.keys())
    counts = [prompt_stats[pid]['count'] for pid in prompt_ids]
    axes[1, 0].bar(prompt_ids, counts, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Prompt ID')
    axes[1, 0].set_ylabel('Usage Count')
    axes[1, 0].set_title('Prompt Usage Frequency')
    axes[1, 0].set_xticks(prompt_ids)
    
    # 4. 每个Prompt的平均Dice
    avg_dices = [prompt_stats[pid]['dice_sum'] / prompt_stats[pid]['count'] 
                 for pid in prompt_ids]
    axes[1, 1].bar(prompt_ids, avg_dices, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Prompt ID')
    axes[1, 1].set_ylabel('Average Dice')
    axes[1, 1].set_title('Average Dice per Prompt')
    axes[1, 1].set_xticks(prompt_ids)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 结果图保存至: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="快速评估Sa2VA Prompt优化RL策略")
    
    parser.add_argument(
        '--rl_model_path',
        type=str,
        required=True,
        help='训练好的RL模型路径（.zip文件）'
    )
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
        default='/home/ubuntu/Sa2VA/rl_prompt_optimization/evaluations',
        help='输出目录'
    )
    parser.add_argument('--max_steps', type=int, default=3, help='每个episode的最大步数')
    parser.add_argument('--max_samples', type=int, default=50, help='评估的最大样本数')
    parser.add_argument('--gpu', type=int, default=2, help='使用的GPU ID')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()
