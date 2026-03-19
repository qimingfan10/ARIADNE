#!/usr/bin/env python
"""
评估序列决策RL Agent

对比实验：
1. RL Agent (PPO-LSTM) vs Statistical Baseline
2. 计算Sensitivity, Precision, F1-Score
3. 生成详细评估报告

使用方法:
    python evaluate_sequential_rl_agent.py --model_path rl_sequential_logs/run_xxx/best_model/best_model.zip
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from rl_sequential_stenosis_env import SequentialStenosisEnv
from stenosis_detection.improved_detectors import detect_stenosis_statistical


class SequentialRLEvaluator:
    """RL Agent评估器"""
    
    def __init__(
        self,
        model_path: str,
        dataset_dir: str,
        mask_dir: str,
        output_dir: str = './rl_evaluation_results'
    ):
        self.model_path = model_path
        self.dataset_dir = Path(dataset_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"加载RL模型: {model_path}")
        self.model = PPO.load(model_path)
        
        # 创建环境（用于获取样本列表）
        self.env = SequentialStenosisEnv(
            dataset_dir=str(self.dataset_dir),
            mask_dir=str(self.mask_dir)
        )
        
        print(f"✅ 模型加载完成")
        print(f"✅ 数据集: {len(self.env.samples)} 个样本")
    
    def evaluate_rl_agent(self, num_samples: int = None) -> Dict:
        """
        评估RL Agent
        
        Args:
            num_samples: 评估样本数量（None表示全部）
        
        Returns:
            评估结果字典
        """
        if num_samples is None:
            num_samples = len(self.env.samples)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_detections = 0
        total_ground_truth = 0
        
        sample_results = []
        
        print(f"\n评估RL Agent...")
        for i in tqdm(range(min(num_samples, len(self.env.samples)))):
            # 设置当前样本
            self.env.current_sample_idx = i
            obs, info = self.env.reset()
            
            # 运行Episode
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            
            # 收集结果
            if 'tp' in info:
                sample_result = {
                    'sample_idx': i,
                    'image_name': Path(self.env.samples[i]['image_path']).name,
                    'tp': info['tp'],
                    'fp': info['fp'],
                    'fn': info['fn'],
                    'num_detections': info['num_detections'],
                    'num_ground_truth': info['num_ground_truth'],
                    'sensitivity': info['sensitivity'],
                    'precision': info['precision'],
                    'f1_score': info['f1_score']
                }
                
                sample_results.append(sample_result)
                
                total_tp += info['tp']
                total_fp += info['fp']
                total_fn += info['fn']
                total_detections += info['num_detections']
                total_ground_truth += info['num_ground_truth']
        
        # 计算整体指标
        overall_sensitivity = total_tp / total_ground_truth if total_ground_truth > 0 else 0.0
        overall_precision = total_tp / total_detections if total_detections > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_sensitivity / 
                     (overall_precision + overall_sensitivity) 
                     if (overall_precision + overall_sensitivity) > 0 else 0.0)
        
        return {
            'method': 'RL Agent (PPO-LSTM)',
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_detections': total_detections,
            'total_ground_truth': total_ground_truth,
            'sensitivity': overall_sensitivity,
            'precision': overall_precision,
            'f1_score': overall_f1,
            'sample_results': sample_results
        }
    
    def evaluate_statistical_baseline(self, num_samples: int = None) -> Dict:
        """
        评估Statistical Baseline
        
        使用Statistical算法作为对比
        """
        if num_samples is None:
            num_samples = len(self.env.samples)
        
        from stenosis_detection.stenosis_detector import StenosisDetector
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_detections = 0
        total_ground_truth = 0
        
        sample_results = []
        
        print(f"\n评估Statistical Baseline...")
        for i in tqdm(range(min(num_samples, len(self.env.samples)))):
            sample = self.env.samples[i]
            
            try:
                # 使用Statistical算法检测
                detector = StenosisDetector(
                    sample['image_path'],
                    sample['mask_path'],
                    resize_shape=(512, 512)
                )
                
                detector.extract_skeleton()
                detector.calculate_vessel_radius(max_radius=110, use_gpu=False)
                detector.find_segmentation_points()
                detector.filter_segmentation_points(min_distance=8)
                detector.detect_stenosis(
                    stenosis_threshold=0.25,
                    min_avg_radius=4
                )
                
                # 获取检测结果
                if hasattr(detector, 'all_stenosis_points') and len(detector.all_stenosis_points) > 0:
                    detections = [(int(p[0]), int(p[1])) for p in detector.all_stenosis_points]
                else:
                    detections = []
                
                # 匹配Ground Truth
                ground_truth = sample['ground_truth']
                tp, fp, fn = self._match_detections(detections, ground_truth)
                
                num_detections = len(detections)
                num_gt = len(ground_truth)
                
                sensitivity = tp / num_gt if num_gt > 0 else 0.0
                precision = tp / num_detections if num_detections > 0 else 0.0
                f1 = (2 * precision * sensitivity / (precision + sensitivity) 
                     if (precision + sensitivity) > 0 else 0.0)
                
                sample_result = {
                    'sample_idx': i,
                    'image_name': Path(sample['image_path']).name,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'num_detections': num_detections,
                    'num_ground_truth': num_gt,
                    'sensitivity': sensitivity,
                    'precision': precision,
                    'f1_score': f1
                }
                
                sample_results.append(sample_result)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_detections += num_detections
                total_ground_truth += num_gt
                
            except Exception as e:
                print(f"⚠ 处理样本 {i} 失败: {e}")
                continue
        
        # 计算整体指标
        overall_sensitivity = total_tp / total_ground_truth if total_ground_truth > 0 else 0.0
        overall_precision = total_tp / total_detections if total_detections > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_sensitivity / 
                     (overall_precision + overall_sensitivity) 
                     if (overall_precision + overall_sensitivity) > 0 else 0.0)
        
        return {
            'method': 'Statistical Baseline',
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_detections': total_detections,
            'total_ground_truth': total_ground_truth,
            'sensitivity': overall_sensitivity,
            'precision': overall_precision,
            'f1_score': overall_f1,
            'sample_results': sample_results
        }
    
    def _match_detections(self, detections, ground_truth, threshold_mm=10.0):
        """匹配检测结果与Ground Truth"""
        threshold_pixels = threshold_mm / self.env.pixel_spacing
        
        gt_matched = [False] * len(ground_truth)
        tp = 0
        fp = 0
        
        for det in detections:
            matched = False
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                dist = np.sqrt((det[0] - gt[0])**2 + (det[1] - gt[1])**2)
                
                if dist <= threshold_pixels:
                    gt_matched[gt_idx] = True
                    tp += 1
                    matched = True
                    break
            
            if not matched:
                fp += 1
        
        fn = len(ground_truth) - tp
        
        return tp, fp, fn
    
    def generate_comparison_report(self, rl_results: Dict, baseline_results: Dict):
        """生成对比报告"""
        report_path = self.output_dir / 'comparison_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("序列决策RL Agent vs Statistical Baseline - 对比评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            # RL Agent结果
            f.write("【RL Agent (PPO-LSTM)】\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ground Truth总数: {rl_results['total_ground_truth']}\n")
            f.write(f"检测总数: {rl_results['total_detections']}\n")
            f.write(f"True Positives: {rl_results['total_tp']}\n")
            f.write(f"False Positives: {rl_results['total_fp']}\n")
            f.write(f"False Negatives: {rl_results['total_fn']}\n\n")
            f.write(f"Sensitivity: {rl_results['sensitivity']:.4f} ({rl_results['sensitivity']*100:.2f}%)\n")
            f.write(f"Precision: {rl_results['precision']:.4f} ({rl_results['precision']*100:.2f}%)\n")
            f.write(f"F1-Score: {rl_results['f1_score']:.4f}\n\n")
            
            # Statistical Baseline结果
            f.write("【Statistical Baseline】\n")
            f.write("-" * 80 + "\n")
            f.write(f"Ground Truth总数: {baseline_results['total_ground_truth']}\n")
            f.write(f"检测总数: {baseline_results['total_detections']}\n")
            f.write(f"True Positives: {baseline_results['total_tp']}\n")
            f.write(f"False Positives: {baseline_results['total_fp']}\n")
            f.write(f"False Negatives: {baseline_results['total_fn']}\n\n")
            f.write(f"Sensitivity: {baseline_results['sensitivity']:.4f} ({baseline_results['sensitivity']*100:.2f}%)\n")
            f.write(f"Precision: {baseline_results['precision']:.4f} ({baseline_results['precision']*100:.2f}%)\n")
            f.write(f"F1-Score: {baseline_results['f1_score']:.4f}\n\n")
            
            # 对比
            f.write("【性能对比】\n")
            f.write("-" * 80 + "\n")
            
            sensitivity_diff = rl_results['sensitivity'] - baseline_results['sensitivity']
            precision_diff = rl_results['precision'] - baseline_results['precision']
            f1_diff = rl_results['f1_score'] - baseline_results['f1_score']
            
            f.write(f"Sensitivity提升: {sensitivity_diff:+.4f} ({sensitivity_diff*100:+.2f}%)\n")
            f.write(f"Precision提升: {precision_diff:+.4f} ({precision_diff*100:+.2f}%)\n")
            f.write(f"F1-Score提升: {f1_diff:+.4f} ({f1_diff*100:+.2f}%)\n")
            
            if f1_diff > 0:
                f.write(f"\n✅ RL Agent性能更优！F1提升 {f1_diff*100:.2f}%\n")
            else:
                f.write(f"\n⚠ Baseline性能更优，F1提升 {-f1_diff*100:.2f}%\n")
        
        print(f"\n✅ 对比报告已保存到: {report_path}")
        
        # 绘制对比图
        self._plot_comparison(rl_results, baseline_results)
    
    def _plot_comparison(self, rl_results: Dict, baseline_results: Dict):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['RL Agent\n(PPO-LSTM)', 'Statistical\nBaseline']
        
        metrics = ['Sensitivity', 'Precision', 'F1-Score']
        rl_values = [
            rl_results['sensitivity'],
            rl_results['precision'],
            rl_results['f1_score']
        ]
        baseline_values = [
            baseline_results['sensitivity'],
            baseline_results['precision'],
            baseline_results['f1_score']
        ]
        
        for i, (metric, rl_val, baseline_val) in enumerate(zip(metrics, rl_values, baseline_values)):
            axes[i].bar(methods, [rl_val, baseline_val], color=['#2ecc71', '#3498db'])
            axes[i].set_ylabel(metric)
            axes[i].set_ylim([0, 1])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].grid(True, alpha=0.3)
            
            # 标注数值
            for j, v in enumerate([rl_val, baseline_val]):
                axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_chart.png', dpi=150)
        print(f"✅ 对比图已保存到: {self.output_dir / 'comparison_chart.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="评估序列决策RL Agent")
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='RL模型路径 (.zip文件)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/mnt/sda1/luoyu/xzjc_data/dataset',
        help='数据集目录'
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='/mnt/sda1/luoyu/xzjc_data/masks',
        help='Mask目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./rl_evaluation_results',
        help='输出目录'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='评估样本数量（默认全部）'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='跳过Baseline评估'
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = SequentialRLEvaluator(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir
    )
    
    # 评估RL Agent
    rl_results = evaluator.evaluate_rl_agent(num_samples=args.num_samples)
    
    print("\n" + "=" * 80)
    print("RL Agent评估结果")
    print("=" * 80)
    print(f"Sensitivity: {rl_results['sensitivity']:.4f}")
    print(f"Precision: {rl_results['precision']:.4f}")
    print(f"F1-Score: {rl_results['f1_score']:.4f}")
    
    # 评估Baseline
    if not args.skip_baseline:
        baseline_results = evaluator.evaluate_statistical_baseline(num_samples=args.num_samples)
        
        print("\n" + "=" * 80)
        print("Statistical Baseline评估结果")
        print("=" * 80)
        print(f"Sensitivity: {baseline_results['sensitivity']:.4f}")
        print(f"Precision: {baseline_results['precision']:.4f}")
        print(f"F1-Score: {baseline_results['f1_score']:.4f}")
        
        # 生成对比报告
        evaluator.generate_comparison_report(rl_results, baseline_results)
    
    print("\n✅ 评估完成！")


if __name__ == '__main__':
    main()
