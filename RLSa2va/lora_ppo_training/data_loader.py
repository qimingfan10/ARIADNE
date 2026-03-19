"""
数据加载
为RL训练准备数据
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random


class VesselSegmentationDataset(Dataset):
    """血管分割数据集"""
    
    def __init__(self, data_root, max_samples=None, split='train', seed=42):
        """
        Args:
            data_root: 数据根目录
            max_samples: 最大样本数
            split: 数据集划分 ('train', 'val', 'test')
            seed: 随机种子
        """
        self.data_root = data_root
        self.split = split
        
        # 加载annotations
        annotations_path = os.path.join(data_root, 'annotations.json')
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # 划分数据集
        random.seed(seed)
        random.shuffle(annotations)
        
        total = len(annotations)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        if split == 'train':
            annotations = annotations[:train_size]
        elif split == 'val':
            annotations = annotations[train_size:train_size + val_size]
        elif split == 'test':
            annotations = annotations[train_size + val_size:]
        
        # 限制样本数
        if max_samples is not None:
            annotations = annotations[:max_samples]
        
        self.annotations = annotations
        self.images_dir = os.path.join(data_root, 'images')
        
        print(f"✅ 加载{split}集: {len(self.annotations)}个样本")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image
            mask: numpy array (H, W)
            image_path: 图像路径
        """
        ann = self.annotations[idx]
        
        # 加载图像
        image_path = os.path.join(self.images_dir, ann['image'])
        image = Image.open(image_path).convert('RGB')
        
        # 生成mask
        width, height = image.size
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        if 'mask' in ann and len(ann['mask']) > 0:
            polygons = ann['mask']
            for polygon in polygons:
                if len(polygon) >= 6:
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, fill=255)
        
        mask = np.array(mask_img)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': ann['image']
        }


def custom_collate_fn(batch):
    """
    自定义collate函数，处理PIL.Image对象
    因为batch_size=1，所以直接返回batch list
    """
    return batch


def create_dataloaders(
    data_root,
    train_samples=None,
    val_samples=None,
    batch_size=1,
    num_workers=4,
    seed=42
):
    """
    创建训练和验证DataLoader
    
    Args:
        data_root: 数据根目录
        train_samples: 训练样本数
        val_samples: 验证样本数
        batch_size: 批大小
        num_workers: 工作线程数
        seed: 随机种子
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = VesselSegmentationDataset(
        data_root, 
        max_samples=train_samples,
        split='train',
        seed=seed
    )
    
    val_dataset = VesselSegmentationDataset(
        data_root,
        max_samples=val_samples,
        split='val',
        seed=seed
    )
    
    # 注意：使用自定义collate_fn处理PIL.Image
    # RL训练通常batch_size=1，因为每个样本需要独立采样
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 自定义collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 自定义collate
    )
    
    return train_loader, val_loader


def prepare_data_split(data_root, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    准备数据划分并保存索引文件
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    # 加载annotations
    annotations_path = os.path.join(data_root, 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # 随机打乱
    random.seed(seed)
    indices = list(range(len(annotations)))
    random.shuffle(indices)
    
    # 划分
    total = len(indices)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    
    split_info = {
        'total_samples': total,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'seed': seed
    }
    
    output_path = os.path.join(output_dir, 'data_split.json')
    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ 数据划分已保存: {output_path}")
    print(f"   训练集: {len(train_indices)}个样本")
    print(f"   验证集: {len(val_indices)}个样本")
    print(f"   测试集: {len(test_indices)}个样本")
    
    return split_info


class DataAugmentation:
    """数据增强（可选）"""
    
    def __init__(self, 
                 horizontal_flip=True,
                 vertical_flip=True,
                 rotation_range=15,
                 brightness_range=0.2,
                 contrast_range=0.2):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image, mask):
        """
        应用数据增强
        
        Args:
            image: PIL Image
            mask: numpy array
        
        Returns:
            augmented_image: PIL Image
            augmented_mask: numpy array
        """
        import torchvision.transforms.functional as TF
        
        # 转换mask为PIL
        mask_pil = Image.fromarray(mask)
        
        # 水平翻转
        if self.horizontal_flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask_pil = TF.hflip(mask_pil)
        
        # 垂直翻转
        if self.vertical_flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask_pil = TF.vflip(mask_pil)
        
        # 旋转
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle)
            mask_pil = TF.rotate(mask_pil, angle)
        
        # 亮度对比度（仅对图像）
        if self.brightness_range > 0:
            brightness_factor = random.uniform(1 - self.brightness_range, 
                                             1 + self.brightness_range)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if self.contrast_range > 0:
            contrast_factor = random.uniform(1 - self.contrast_range,
                                           1 + self.contrast_range)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # 转回numpy
        mask = np.array(mask_pil)
        
        return image, mask
