"""
奖励函数：多目标优化
目标：Dice + Recall + 拓扑连通性 + 长度约束
"""

import numpy as np
import torch
from scipy.ndimage import label
from skimage.morphology import skeletonize, thin
from skimage.measure import label as sk_label, regionprops
import cv2


class MultiObjectiveReward:
    """多目标奖励函数"""
    
    def __init__(self, 
                 dice_weight=0.5,
                 recall_weight=0.2,
                 topology_weight=0.2,
                 length_weight=0.1,
                 recall_target=0.85,
                 verbose=False):
        """
        Args:
            dice_weight: Dice分数权重
            recall_weight: Recall奖励权重
            topology_weight: 拓扑连通性权重
            length_weight: 长度惩罚权重
            recall_target: Recall目标值
            verbose: 是否输出详细日志
        """
        self.dice_weight = dice_weight
        self.recall_weight = recall_weight
        self.topology_weight = topology_weight
        self.length_weight = length_weight
        self.recall_target = recall_target
        self.verbose = verbose
        
        # 归一化权重
        total = dice_weight + recall_weight + topology_weight + length_weight
        self.dice_weight /= total
        self.recall_weight /= total
        self.topology_weight /= total
        self.length_weight /= total
    
    def __call__(self, pred_mask, gt_mask):
        """
        计算综合奖励
        
        Args:
            pred_mask: 预测mask (H, W)，值为0或1
            gt_mask: GT mask (H, W)，值为0或1
            
        Returns:
            total_reward: 总奖励（标量）
            reward_dict: 各项奖励的详细信息
        """
        # 确保是numpy数组
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        # 二值化
        pred = (pred_mask > 0).astype(np.float32)
        gt = (gt_mask > 0).astype(np.float32)
        
        reward_dict = {}
        
        # 1. Dice Score奖励
        dice_score = self._compute_dice(pred, gt)
        dice_reward = dice_score * 10.0  # Scale到0-10
        reward_dict['dice'] = dice_score
        reward_dict['dice_reward'] = dice_reward
        
        # 2. Recall奖励（针对性优化）
        recall = self._compute_recall(pred, gt)
        if recall < self.recall_target:
            # Recall低于目标，给予负奖励
            recall_reward = (recall - self.recall_target) * 20.0
        else:
            # Recall达标，给予小额正奖励
            recall_reward = (recall - self.recall_target) * 5.0
        reward_dict['recall'] = recall
        reward_dict['recall_reward'] = recall_reward
        
        # 3. 拓扑连通性奖励
        topology_score = self._compute_topology(pred, gt)
        topology_reward = topology_score * 5.0
        reward_dict['topology_score'] = topology_score
        reward_dict['topology_reward'] = topology_reward
        
        # 4. 长度约束
        length_ratio = self._compute_length_ratio(pred, gt)
        length_penalty = -abs(1.0 - length_ratio) * 5.0
        reward_dict['length_ratio'] = length_ratio
        reward_dict['length_penalty'] = length_penalty
        
        # 加权求和
        total_reward = (
            self.dice_weight * dice_reward +
            self.recall_weight * recall_reward +
            self.topology_weight * topology_reward +
            self.length_weight * length_penalty
        )
        
        reward_dict['total_reward'] = total_reward
        
        # Precision用于监控
        precision = self._compute_precision(pred, gt)
        reward_dict['precision'] = precision
        
        if self.verbose:
            print(f"Rewards: Total={total_reward:.4f}, "
                  f"Dice={dice_score:.4f}, Recall={recall:.4f}, "
                  f"Topology={topology_score:.4f}, Length={length_ratio:.4f}")
        
        return total_reward, reward_dict
    
    def _compute_dice(self, pred, gt):
        """计算Dice分数"""
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        return float(dice)
    
    def _compute_recall(self, pred, gt):
        """计算Recall（敏感度）"""
        intersection = (pred * gt).sum()
        recall = intersection / (gt.sum() + 1e-8)
        return float(recall)
    
    def _compute_precision(self, pred, gt):
        """计算Precision"""
        intersection = (pred * gt).sum()
        precision = intersection / (pred.sum() + 1e-8)
        return float(precision)
    
    def _compute_topology(self, pred, gt):
        """
        计算拓扑连通性得分
        考虑：连通分量数量、交叉点（分叉）、端点
        """
        if pred.sum() < 10:  # 预测几乎为空
            return -1.0
        
        try:
            # 骨架化
            pred_skel = skeletonize(pred > 0)
            gt_skel = skeletonize(gt > 0)
            
            # 1. 连通分量数量（越接近GT越好）
            pred_components = sk_label(pred_skel, connectivity=2)
            gt_components = sk_label(gt_skel, connectivity=2)
            n_pred = pred_components.max()
            n_gt = gt_components.max()
            
            # 惩罚过多的断裂
            component_score = 1.0 - min(abs(n_pred - n_gt) / max(n_gt, 1), 1.0)
            
            # 2. 交叉点（血管分叉）
            pred_junctions = self._count_junctions(pred_skel)
            gt_junctions = self._count_junctions(gt_skel)
            
            if gt_junctions > 0:
                junction_score = min(pred_junctions / gt_junctions, 1.0)
            else:
                junction_score = 1.0 if pred_junctions == 0 else 0.5
            
            # 3. 端点数量（应该接近GT）
            pred_endpoints = self._count_endpoints(pred_skel)
            gt_endpoints = self._count_endpoints(gt_skel)
            
            if gt_endpoints > 0:
                endpoint_score = 1.0 - min(abs(pred_endpoints - gt_endpoints) / gt_endpoints, 1.0)
            else:
                endpoint_score = 1.0 if pred_endpoints == 0 else 0.5
            
            # 综合得分
            topology_score = (
                0.4 * component_score +
                0.4 * junction_score +
                0.2 * endpoint_score
            )
            
            return float(topology_score)
            
        except Exception as e:
            # 骨架化失败，返回较低的分数
            return 0.0
    
    def _count_junctions(self, skeleton):
        """计算骨架的交叉点数量"""
        # 使用3x3邻域卷积核
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # 卷积
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # 交叉点：中心为1，且邻居数>=3
        junctions = (skeleton > 0) & (neighbor_count >= 13)  # 10(中心) + 3(邻居)
        
        return int(junctions.sum())
    
    def _count_endpoints(self, skeleton):
        """计算骨架的端点数量"""
        # 使用3x3邻域卷积核
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # 卷积
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # 端点：中心为1，且邻居数==1
        endpoints = (skeleton > 0) & (neighbor_count == 11)  # 10(中心) + 1(邻居)
        
        return int(endpoints.sum())
    
    def _compute_length_ratio(self, pred, gt):
        """计算血管长度比例"""
        try:
            # 骨架化
            pred_skel = skeletonize(pred > 0)
            gt_skel = skeletonize(gt > 0)
            
            pred_length = pred_skel.sum()
            gt_length = gt_skel.sum()
            
            if gt_length == 0:
                return 1.0 if pred_length == 0 else 0.0
            
            ratio = pred_length / gt_length
            return float(ratio)
            
        except Exception as e:
            return 1.0


class SimpleDiceReward:
    """简化版：仅使用Dice作为奖励"""
    
    def __call__(self, pred_mask, gt_mask):
        """计算Dice奖励"""
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        pred = (pred_mask > 0).astype(np.float32)
        gt = (gt_mask > 0).astype(np.float32)
        
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        
        # Scale到0-10
        reward = dice * 10.0
        
        reward_dict = {
            'dice': float(dice),
            'total_reward': float(reward)
        }
        
        return reward, reward_dict


class RecallFocusedReward:
    """专注于提升Recall的奖励函数"""
    
    def __init__(self, recall_target=0.85, recall_weight=0.7):
        self.recall_target = recall_target
        self.recall_weight = recall_weight
        self.dice_weight = 1.0 - recall_weight
    
    def __call__(self, pred_mask, gt_mask):
        """计算以Recall为主的奖励"""
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        pred = (pred_mask > 0).astype(np.float32)
        gt = (gt_mask > 0).astype(np.float32)
        
        # Dice
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        
        # Recall
        recall = intersection / (gt.sum() + 1e-8)
        
        # Recall奖励：低于目标时大幅惩罚
        if recall < self.recall_target:
            recall_reward = (recall - self.recall_target) * 30.0
        else:
            recall_reward = (recall - self.recall_target) * 10.0
        
        # 综合奖励
        total_reward = (
            self.dice_weight * dice * 10.0 +
            self.recall_weight * recall_reward
        )
        
        reward_dict = {
            'dice': float(dice),
            'recall': float(recall),
            'precision': float(intersection / (pred.sum() + 1e-8)),
            'total_reward': float(total_reward)
        }
        
        return total_reward, reward_dict


# 默认使用多目标奖励
def get_reward_function(reward_type='multi_objective', **kwargs):
    """获取奖励函数"""
    if reward_type == 'multi_objective':
        return MultiObjectiveReward(**kwargs)
    elif reward_type == 'simple_dice':
        return SimpleDiceReward()
    elif reward_type == 'recall_focused':
        return RecallFocusedReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
