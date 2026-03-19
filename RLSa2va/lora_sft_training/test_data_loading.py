"""
测试数据加载是否正常
"""
import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("测试数据加载...")

# 数据路径
data_root = '/home/ubuntu/Sa2VA/Segment_DATA_Merged_512'
images_dir = os.path.join(data_root, 'images')
masks_dir = os.path.join(data_root, 'masks')

# 获取图像列表
all_images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
print(f"\n总图像数: {len(all_images)}")

# 训练/验证划分
train_ratio = 0.8
n_train = int(len(all_images) * train_ratio)
train_images = all_images[:n_train]
val_images = all_images[n_train:]

print(f"训练集: {len(train_images)}")
print(f"验证集: {len(val_images)}")

# 加载第一个样本
image_path = train_images[0]
img_name = os.path.basename(image_path)
mask_name = img_name.replace('.jpg', '_mask.png')
mask_path = os.path.join(masks_dir, mask_name)

print(f"\n测试样本:")
print(f"  图像: {img_name}")
print(f"  Mask: {mask_name}")

# 加载
image = Image.open(image_path).convert('RGB')
mask = Image.open(mask_path).convert('L')
mask = np.array(mask).astype(np.float32) / 255.0
mask = (mask > 0.5).astype(np.float32)

sample = {'image': image, 'mask': mask, 'image_name': img_name}

print(f"\n样本信息:")
print(f"  图像: {sample['image'].size}")
print(f"  Mask shape: {sample['mask'].shape}")
print(f"  Mask范围: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
print(f"  Mask唯一值: {np.unique(sample['mask'])}")
print(f"  血管像素占比: {sample['mask'].mean():.4f} ({sample['mask'].mean()*100:.2f}%)")
print(f"  文件名: {sample['image_name']}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(sample['image'])
axes[0].set_title('原图')
axes[0].axis('off')

axes[1].imshow(sample['mask'], cmap='gray')
axes[1].set_title('GT Mask')
axes[1].axis('off')

plt.tight_layout()
output_path = '/home/ubuntu/Sa2VA/lora_sft_training/data_sample.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化保存至: {output_path}")

print("\n" + "="*60)
print("✅ 数据加载测试通过！")
print("="*60)
