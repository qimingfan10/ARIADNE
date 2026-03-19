"""
评估狭窄检测算法的准确率

在标注数据集上测试不同算法，计算Precision, Recall, F1-Score
"""
import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from collections import defaultdict
import time
import warnings

# 添加路径
sys.path.insert(0, '/mnt/sda1/luoyu/SAM-VMNet')

from stenosis_detection.stenosis_detector import StenosisDetector, find_path, get_radius
from stenosis_detection.improved_detectors import ALGORITHMS


def load_ground_truth(xml_path):
    """
    从XML文件加载ground truth标注
    
    返回:
        bboxes: [(xmin, ymin, xmax, ymax), ...]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        bboxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'Stenosis':
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bboxes.append((xmin, ymin, xmax, ymax))
        
        return bboxes
    except Exception as e:
        warnings.warn(f"Failed to load {xml_path}: {e}")
        return []


def point_in_bbox(point, bbox, tolerance=15):
    """
    检查点是否在bbox内（带容差）
    
    参数:
        point: (x, y)
        bbox: (xmin, ymin, xmax, ymax)
        tolerance: 允许的像素偏差
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    
    return (xmin - tolerance <= x <= xmax + tolerance and
            ymin - tolerance <= y <= ymax + tolerance)


def calculate_metrics(detected_points, ground_truth_bboxes, tolerance=15):
    """
    计算检测指标
    
    返回:
        dict: {
            'precision': float,
            'recall': float,
            'f1_score': float,
            'tp': int,
            'fp': int,
            'fn': int
        }
    """
    if len(ground_truth_bboxes) == 0:
        if len(detected_points) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 
                    'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0,
                    'tp': 0, 'fp': len(detected_points), 'fn': 0}
    
    if len(detected_points) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'tp': 0, 'fp': 0, 'fn': len(ground_truth_bboxes)}
    
    # 匹配检测点到ground truth
    matched_gt = set()
    true_positives = 0
    
    for det_point in detected_points:
        matched = False
        for idx, gt_bbox in enumerate(ground_truth_bboxes):
            if idx not in matched_gt:
                if point_in_bbox(det_point, gt_bbox, tolerance):
                    matched = True
                    matched_gt.add(idx)
                    true_positives += 1
                    break
    
    false_positives = len(detected_points) - true_positives
    false_negatives = len(ground_truth_bboxes) - len(matched_gt)
    
    # 计算指标
    precision = true_positives / len(detected_points) if len(detected_points) > 0 else 0
    recall = true_positives / len(ground_truth_bboxes) if len(ground_truth_bboxes) > 0 else 0
    
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }


def run_detection_with_algorithm(image_path, mask_path, algorithm_func):
    """
    使用指定算法运行检测
    
    返回:
        detected_points: [(x, y), ...]
    """
    try:
        # StenosisDetector需要图像路径，如果没有mask路径，使用图像路径
        if not mask_path or not os.path.exists(mask_path):
            mask_path = image_path  # 使用相同图像作为mask
        
        # 创建检测器（传入路径）
        detector = StenosisDetector(image_path, mask_path)
        
        # 运行基础步骤
        detector.extract_skeleton()
        detector.calculate_vessel_radius(use_gpu=False)
        detector.find_segmentation_points()
        detector.filter_segmentation_points()
        
        # 使用指定算法检测狭窄
        all_stenosis_points = []
        
        if len(detector.final_segmentation_points) > 1:
            for i in range(len(detector.final_segmentation_points) - 1):
                start_point = detector.final_segmentation_points[i]
                end_point = detector.final_segmentation_points[i + 1]
                
                try:
                    # 查找路径
                    shortest_path, path_length = find_path(
                        detector.skeleton, start_point, end_point
                    )
                    
                    # 获取路径上所有点的半径
                    path_points = []
                    path_radii = []
                    for k in range(path_length):
                        pt = shortest_path[k]
                        r = get_radius(detector.point_data, pt)
                        path_points.append(tuple(pt))
                        path_radii.append(r)
                    
                    if len(path_radii) > 0:
                        avg_r = np.mean(path_radii)
                        
                        # 使用指定算法检测
                        stenosis_pts, _ = algorithm_func(
                            path_points,
                            path_radii,
                            avg_r
                        )
                        
                        all_stenosis_points.extend(stenosis_pts)
                
                except Exception as e:
                    continue
        
        return all_stenosis_points
        
    except Exception as e:
        warnings.warn(f"Detection failed for {image_path}: {e}")
        return []


def evaluate_algorithm(algorithm_name, algorithm_func, test_images, tolerance=15):
    """
    评估单个算法
    
    返回:
        results: dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"评估算法: {algorithm_name}")
    print(f"{'='*60}")
    
    all_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    successful_tests = 0
    failed_tests = 0
    
    start_time = time.time()
    
    for idx, (img_path, xml_path) in enumerate(test_images):
        if (idx + 1) % 10 == 0:
            print(f"进度: {idx + 1}/{len(test_images)}...")
        
        try:
            # 加载ground truth
            gt_bboxes = load_ground_truth(xml_path)
            if len(gt_bboxes) == 0:
                continue
            
            # 运行检测
            mask_path = None  # 使用图像自动生成mask
            detected_points = run_detection_with_algorithm(
                img_path, mask_path, algorithm_func
            )
            
            # 计算指标
            metrics = calculate_metrics(detected_points, gt_bboxes, tolerance)
            all_metrics.append(metrics)
            
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            
            successful_tests += 1
            
        except Exception as e:
            failed_tests += 1
            continue
    
    elapsed_time = time.time() - start_time
    
    # 汇总结果
    if successful_tests == 0:
        return {
            'algorithm': algorithm_name,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
            'successful_tests': 0,
            'failed_tests': failed_tests,
            'time': elapsed_time
        }
    
    # 按图像平均
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
    
    # 总体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / 
                  (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0
    
    results = {
        'algorithm': algorithm_name,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'time': elapsed_time
    }
    
    # 打印结果
    print(f"\n结果:")
    print(f"  成功测试: {successful_tests}/{successful_tests + failed_tests}")
    print(f"  平均 Precision: {avg_precision:.3f}")
    print(f"  平均 Recall: {avg_recall:.3f}")
    print(f"  平均 F1-Score: {avg_f1:.3f}")
    print(f"  总体 Precision: {overall_precision:.3f}")
    print(f"  总体 Recall: {overall_recall:.3f}")
    print(f"  总体 F1-Score: {overall_f1:.3f}")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  用时: {elapsed_time:.1f}秒")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估狭窄检测算法准确率')
    parser.add_argument('--num-images', type=int, default=100,
                        help='测试图像数量 (default: 100)')
    parser.add_argument('--dataset-dir', type=str, 
                        default='/mnt/sda1/luoyu/xzjc_data/dataset',
                        help='数据集目录')
    parser.add_argument('--tolerance', type=int, default=15,
                        help='匹配容差 (pixels)')
    parser.add_argument('--algorithms', nargs='+', 
                        default=['baseline', 'statistical', 'signal_processing', 'multi_pattern'],
                        help='要测试的算法')
    parser.add_argument('--output', type=str, 
                        default='accuracy_comparison_report.md',
                        help='输出报告文件')
    
    args = parser.parse_args()
    
    # 获取测试图像
    dataset_dir = args.dataset_dir
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.bmp')]
    
    # 随机选择
    np.random.seed(42)
    test_files = np.random.choice(all_files, 
                                   min(args.num_images, len(all_files)), 
                                   replace=False)
    
    test_images = []
    for f in test_files:
        img_path = os.path.join(dataset_dir, f)
        xml_path = os.path.join(dataset_dir, f.replace('.bmp', '.xml'))
        if os.path.exists(xml_path):
            test_images.append((img_path, xml_path))
    
    print(f"\n{'='*60}")
    print(f"狭窄检测准确率评估")
    print(f"{'='*60}")
    print(f"数据集: {dataset_dir}")
    print(f"测试图像: {len(test_images)}")
    print(f"匹配容差: {args.tolerance} pixels")
    print(f"测试算法: {args.algorithms}")
    
    # 评估所有算法
    all_results = []
    for algo_name in args.algorithms:
        if algo_name not in ALGORITHMS:
            print(f"警告: 未知算法 {algo_name}，跳过")
            continue
        
        algo_func = ALGORITHMS[algo_name]
        results = evaluate_algorithm(algo_name, algo_func, test_images, args.tolerance)
        all_results.append(results)
    
    # 生成对比报告
    print(f"\n{'='*60}")
    print("总结对比")
    print(f"{'='*60}")
    print(f"{'算法':<20} {'P(avg)':<10} {'R(avg)':<10} {'F1(avg)':<10} {'F1(total)':<10}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['algorithm']:<20} "
              f"{r['avg_precision']:<10.3f} "
              f"{r['avg_recall']:<10.3f} "
              f"{r['avg_f1_score']:<10.3f} "
              f"{r['overall_f1_score']:<10.3f}")
    
    # 找最佳算法
    best_algo = max(all_results, key=lambda x: x['overall_f1_score'])
    print(f"\n最佳算法: {best_algo['algorithm']} (F1={best_algo['overall_f1_score']:.3f})")
    
    # 保存报告
    save_report(all_results, args.output, test_images, args.tolerance)
    print(f"\n报告已保存到: {args.output}")


def save_report(results, output_path, test_images, tolerance):
    """保存详细报告到Markdown文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 狭窄检测准确率对比报告\n\n")
        
        f.write("## 测试配置\n\n")
        f.write(f"- **测试图像数**: {len(test_images)}\n")
        f.write(f"- **匹配容差**: {tolerance} pixels\n")
        f.write(f"- **数据集**: xzjc_data\n\n")
        
        f.write("## 总体对比\n\n")
        f.write("| 算法 | Precision (平均) | Recall (平均) | F1-Score (平均) | F1-Score (总体) | TP | FP | FN | 用时(秒) |\n")
        f.write("|------|------------------|---------------|-----------------|-----------------|----|----|----|---------|\n")
        
        for r in results:
            f.write(f"| {r['algorithm']} | "
                   f"{r['avg_precision']:.3f} | "
                   f"{r['avg_recall']:.3f} | "
                   f"{r['avg_f1_score']:.3f} | "
                   f"{r['overall_f1_score']:.3f} | "
                   f"{r['total_tp']} | "
                   f"{r['total_fp']} | "
                   f"{r['total_fn']} | "
                   f"{r['time']:.1f} |\n")
        
        f.write("\n## 详细分析\n\n")
        
        # 找baseline
        baseline = next((r for r in results if r['algorithm'] == 'baseline'), None)
        
        for r in results:
            f.write(f"### {r['algorithm']}\n\n")
            f.write(f"**性能指标**:\n")
            f.write(f"- Precision (平均): {r['avg_precision']:.3f}\n")
            f.write(f"- Recall (平均): {r['avg_recall']:.3f}\n")
            f.write(f"- F1-Score (平均): {r['avg_f1_score']:.3f}\n")
            f.write(f"- F1-Score (总体): {r['overall_f1_score']:.3f}\n")
            f.write(f"- True Positives: {r['total_tp']}\n")
            f.write(f"- False Positives: {r['total_fp']}\n")
            f.write(f"- False Negatives: {r['total_fn']}\n")
            f.write(f"- 成功测试: {r['successful_tests']}\n")
            f.write(f"- 用时: {r['time']:.1f}秒\n\n")
            
            if baseline and r['algorithm'] != 'baseline':
                if baseline['overall_f1_score'] > 0:
                    f1_improvement = (r['overall_f1_score'] - baseline['overall_f1_score']) / baseline['overall_f1_score'] * 100
                    f.write(f"**相对Baseline提升**: {f1_improvement:+.1f}%\n\n")
                else:
                    abs_improvement = r['overall_f1_score'] - baseline['overall_f1_score']
                    f.write(f"**绝对提升**: {abs_improvement:+.3f}\n\n")
        
        f.write("\n## 结论\n\n")
        best = max(results, key=lambda x: x['overall_f1_score'])
        f.write(f"**最佳算法**: {best['algorithm']}\n")
        f.write(f"- F1-Score: {best['overall_f1_score']:.3f}\n")
        if baseline:
            if baseline['overall_f1_score'] > 0:
                improvement = (best['overall_f1_score'] - baseline['overall_f1_score']) / baseline['overall_f1_score'] * 100
                f.write(f"- 相对Baseline提升: {improvement:+.1f}%\n")
            else:
                abs_improvement = best['overall_f1_score'] - baseline['overall_f1_score']
                f.write(f"- 绝对提升: {abs_improvement:+.3f}\n")


if __name__ == '__main__':
    main()
