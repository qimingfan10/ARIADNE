"""
测试血管狭窄检测功能
从 Segment_DATA_Merged_512 数据集中随机选择10张图片进行测试
"""

import os
import sys
import random
import glob
from pathlib import Path

# 添加stenosis_detection模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stenosis_detection'))

from stenosis_detection import StenosisDetector


def find_matching_pairs(images_dir, masks_dir, num_samples=10):
    """
    查找匹配的原始图像和mask图像对
    
    参数:
        images_dir: 原始图像目录
        masks_dir: mask图像目录
        num_samples: 需要的样本数量
        
    返回:
        list: [(original_path, mask_path), ...] 图像对列表
    """
    # 获取所有mask文件
    mask_files = glob.glob(os.path.join(masks_dir, "*_mask.png"))
    
    if not mask_files:
        print(f"错误: 在 {masks_dir} 中未找到mask文件")
        return []
    
    print(f"找到 {len(mask_files)} 个mask文件")
    
    # 查找匹配的图像对
    pairs = []
    for mask_path in mask_files:
        mask_name = os.path.basename(mask_path)
        # 移除 _mask.png 后缀得到原始文件名
        original_name = mask_name.replace('_mask.png', '.jpg')
        original_path = os.path.join(images_dir, original_name)
        
        if os.path.exists(original_path):
            pairs.append((original_path, mask_path))
    
    print(f"找到 {len(pairs)} 对匹配的图像")
    
    # 随机选择指定数量的样本
    if len(pairs) > num_samples:
        pairs = random.sample(pairs, num_samples)
        print(f"随机选择 {num_samples} 对图像进行测试")
    
    return pairs


def test_single_image(original_path, mask_path, output_dir, index):
    """
    测试单张图像的狭窄检测
    
    参数:
        original_path: 原始图像路径
        mask_path: mask图像路径
        output_dir: 输出目录
        index: 图像索引
        
    返回:
        dict: 测试结果
    """
    print(f"\n{'='*70}")
    print(f"测试图像 {index}")
    print(f"{'='*70}")
    print(f"原始图像: {os.path.basename(original_path)}")
    print(f"Mask图像: {os.path.basename(mask_path)}")
    
    result = {
        'index': index,
        'original_path': original_path,
        'mask_path': mask_path,
        'success': False,
        'error': None,
        'report': None
    }
    
    try:
        # 创建检测器
        detector = StenosisDetector(
            original_image_path=original_path,
            segmented_image_path=mask_path,
            resize_shape=(512, 512)  # 使用512x512以匹配数据集
        )
        
        # 提取骨架
        detector.extract_skeleton()
        
        # 计算半径（使用CPU，速度更快）
        detector.calculate_vessel_radius(use_gpu=False)
        
        # 运行完整检测
        report = detector.run_full_detection()
        
        # 保存结果图像
        output_filename = f"test_{index:02d}_{os.path.basename(original_path)}"
        output_path = os.path.join(output_dir, output_filename)
        detector.visualize_stenosis_results(save_path=output_path)
        
        result['success'] = True
        result['report'] = report
        result['output_path'] = output_path
        
        print(f"\n✓ 检测成功!")
        print(f"  - 检测到 {report['total_stenosis_points']} 个狭窄点")
        print(f"  - 结果已保存至: {output_path}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"\n✗ 检测失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return result


def main():
    """主测试函数"""
    print("="*70)
    print("血管狭窄检测功能测试")
    print("="*70)
    
    # 设置路径
    data_root = "/mnt/sda1/luoyu/Segment_DATA_Merged_512"
    images_dir = os.path.join(data_root, "images")
    masks_dir = os.path.join(data_root, "masks")
    output_dir = "/mnt/sda1/luoyu/SAM-VMNet/test_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 查找图像对
    print(f"\n正在查找图像对...")
    pairs = find_matching_pairs(images_dir, masks_dir, num_samples=10)
    
    if not pairs:
        print("错误: 未找到匹配的图像对")
        return
    
    # 测试每对图像
    results = []
    for i, (original_path, mask_path) in enumerate(pairs, 1):
        result = test_single_image(original_path, mask_path, output_dir, i)
        results.append(result)
    
    # 生成测试报告
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print(f"\n总测试数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    
    # 详细结果
    print(f"\n{'='*70}")
    print("详细结果")
    print(f"{'='*70}")
    
    for result in results:
        print(f"\n[{result['index']}] {os.path.basename(result['original_path'])}")
        if result['success']:
            report = result['report']
            print(f"  ✓ 成功 - 检测到 {report['total_stenosis_points']} 个狭窄点")
            
            # 显示狭窄详情
            if report['total_stenosis_points'] > 0:
                for detail in report['stenosis_details']:
                    print(f"    - {detail['severity']}: {detail['degree_percentage']}")
        else:
            print(f"  ✗ 失败 - {result['error']}")
    
    # 统计狭窄检测结果
    print(f"\n{'='*70}")
    print("狭窄检测统计")
    print(f"{'='*70}")
    
    total_stenosis = 0
    severity_counts = {'轻度狭窄': 0, '中度狭窄': 0, '重度狭窄': 0}
    
    for result in results:
        if result['success'] and result['report']:
            total_stenosis += result['report']['total_stenosis_points']
            for detail in result['report']['stenosis_details']:
                severity_counts[detail['severity']] += 1
    
    print(f"\n总狭窄点数: {total_stenosis}")
    print(f"轻度狭窄 (25%-50%): {severity_counts['轻度狭窄']}")
    print(f"中度狭窄 (50%-75%): {severity_counts['中度狭窄']}")
    print(f"重度狭窄 (>75%): {severity_counts['重度狭窄']}")
    
    print(f"\n{'='*70}")
    print("测试完成!")
    print(f"{'='*70}")
    print(f"\n所有结果已保存至: {output_dir}")
    
    return results


if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    random.seed(42)
    
    results = main()
