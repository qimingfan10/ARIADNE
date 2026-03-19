"""
组合损失函数：Dice Loss + Focal Loss + BCE Loss
专门用于不平衡分割任务（血管分割）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    """
    组合损失函数
    
    Args:
        weight_dice: Dice Loss权重（关注重叠度）
        weight_focal: Focal Loss权重（关注难样本）
        weight_bce: BCE Loss权重（基础像素分类）
        smooth: 数值稳定性平滑项
    """
    def __init__(self, weight_dice=1.0, weight_focal=1.0, weight_bce=0.5, smooth=1e-5):
        super(ComboLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_bce = weight_bce
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出的logits（未经sigmoid），shape: (B, H, W) or (B, 1, H, W)
            targets: GT mask，值域[0, 1]，shape: (B, H, W) or (B, 1, H, W)
            
        Returns:
            total_loss: 总损失
            dice_score: Dice分数（用于监控）
        """
        # 确保维度一致
        if inputs.dim() == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
            
        # 1. BCE Loss (基础像素分类)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='mean'
        )
        
        # 准备概率图
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        # 2. Dice Loss (关注重叠度)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / \
                     (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # 3. Focal Loss (关注难样本)
        # alpha=0.8: 更关注正样本（血管）
        # gamma=2.0: 标准设置，降低易分样本的权重
        alpha = 0.8
        gamma = 2.0
        
        # pt: 模型对真实类别的预测概率
        pt = torch.where(targets_flat == 1, inputs_flat, 1 - inputs_flat)
        
        # Focal Loss公式
        focal_loss = -alpha * (1 - pt) ** gamma * torch.log(pt + self.smooth)
        focal_loss = focal_loss.mean()
        
        # 组合损失
        total_loss = (self.weight_dice * dice_loss) + \
                     (self.weight_focal * focal_loss) + \
                     (self.weight_bce * bce_loss)
        
        # 返回损失和Dice分数（用于日志）
        return total_loss, dice_score.item(), {
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'bce_loss': bce_loss.item()
        }


class DiceLoss(nn.Module):
    """单独的Dice Loss（如果只想用它）"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / \
                     (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice_score, dice_score.item()


class FocalLoss(nn.Module):
    """单独的Focal Loss（如果只想用它）"""
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        pt = torch.where(targets_flat == 1, inputs_flat, 1 - inputs_flat)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * \
                     torch.log(pt + self.smooth)
        
        return focal_loss.mean()


def calculate_metrics(pred_prob, target):
    """
    计算评估指标
    
    Args:
        pred_prob: 预测概率图，shape: (B, H, W)，值域[0, 1]
        target: GT mask，shape: (B, H, W)，值域[0, 1]
        
    Returns:
        dict: 包含dice, iou, recall, precision
    """
    pred = (pred_prob > 0.5).float()
    target = (target > 0.5).float()
    
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    dice = (2 * intersection + 1e-5) / (pred_flat.sum() + target_flat.sum() + 1e-5)
    iou = (intersection + 1e-5) / (union + 1e-5)
    recall = (intersection + 1e-5) / (target_flat.sum() + 1e-5)
    precision = (intersection + 1e-5) / (pred_flat.sum() + 1e-5)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'recall': recall.item(),
        'precision': precision.item()
    }


if __name__ == '__main__':
    # 测试ComboLoss
    print("测试ComboLoss...")
    
    # 模拟数据
    batch_size = 4
    height, width = 512, 512
    
    # 模拟logits和GT
    logits = torch.randn(batch_size, height, width)
    gt_mask = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # 创建loss
    criterion = ComboLoss(weight_dice=1.0, weight_focal=1.0, weight_bce=0.5)
    
    # 计算loss
    loss, dice_score, loss_dict = criterion(logits, gt_mask)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Dice Score: {dice_score:.4f}")
    print(f"Loss Details: {loss_dict}")
    
    # 测试metrics
    pred_prob = torch.sigmoid(logits)
    metrics = calculate_metrics(pred_prob, gt_mask)
    print(f"Metrics: {metrics}")
    
    print("\n✅ ComboLoss测试通过！")
