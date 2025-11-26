"""
Set prediction loss using Hungarian matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Matching algorithm options (benchmarked on batch_size=128):
# 1. FastHungarianMatcher (RECOMMENDED): 10.5ms, exact, 3.8x faster than original
from .hungarian_matching import FastHungarianMatcher as HungarianMatcher, get_matched_pairs
# 2. HungarianMatcher (Original): 39.7ms, exact but slow (multiple CPU-GPU transfers)
# from .hungarian_matching import HungarianMatcher, get_matched_pairs
# 3. GreedyMatcher: 278ms, slow + only 74.7% match quality (NOT recommended)
# from .hungarian_matching_greedy import GreedyMatcher as HungarianMatcher, get_matched_pairs

import math

class CosineAnnealer:
    def __init__(self, start_value, end_value, total_steps):
        """
        参数:
            start_value: 初始值 (例如 0.01)
            end_value: 最终值 (例如 0.0005)
            total_steps: 总步数 (例如 epochs * len(dataloader))
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_value(self):
        # 如果已经跑完了，就保持在最终值
        if self.current_step >= self.total_steps:
            return self.end_value
        
        # 余弦退火公式
        cosine_val = 0.5 * (1 + math.cos(math.pi * self.current_step / self.total_steps))
        value = self.end_value + (self.start_value - self.end_value) * cosine_val
        
        return value

    def step(self):
        """每次调用，步数+1，并返回当前值"""
        val = self.get_value()
        self.current_step += 1
        return val


def new_scalar_contrastive_loss(pred_mz, gt_mz, indices, temperature=0.01):
    """
    加速版：移除 Python 循环，利用 GPU 并行计算。
    """
    device = pred_mz.device
    batch_size = pred_mz.shape[0]

    # --- 1. 预处理索引 (这是唯一保留的轻量循环，只用于构建索引) ---
    # 我们需要构建全局的索引，把 batch 中所有匹配的 pairs 收集起来
    batch_idx_list = []
    row_idx_list = []
    col_idx_list = []

    # 这一步非常快，因为只处理整数索引，不涉及矩阵运算
    for b, (row_ind, col_ind) in enumerate(indices):
        # row_ind, col_ind 可能是 numpy 数组或 tensor，统一转 tensor
        r = torch.as_tensor(row_ind, device=device, dtype=torch.long)
        c = torch.as_tensor(col_ind, device=device, dtype=torch.long)
        
        # 创建对应的 batch 索引 (例如: [0, 0, 0, 1, 1...])
        batch_idx_list.append(torch.full_like(r, b))
        row_idx_list.append(r)
        col_idx_list.append(c)

    # 拼成一维的长向量 (Total_Matches, )
    if not batch_idx_list: # 防止空列表报错
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    b_idx = torch.cat(batch_idx_list)
    row_idx = torch.cat(row_idx_list)
    target_labels = torch.cat(col_idx_list)

    # --- 2. 批量提取数据 (Gather) ---
    
    # 取出所有匹配上的 Anchor (预测值)
    # pred_mz: [B, N_pred] -> selected: [Total_Matches]
    # b_idx 和 row_idx 共同定位具体的标量值
    anchors = pred_mz[b_idx, row_idx].unsqueeze(1) # [Total_Matches, 1]

    # 取出对应的 Ground Truth 集合
    # 这一步很关键：每个 anchor 都要和它所属那张图的**所有** GT 比较
    # gt_mz: [B, N_gt] -> selected: [Total_Matches, N_gt]
    # 使用 b_idx 索引，会自动把对应的 GT 行复制给该 Batch 下所有的 anchor
    targets_set = gt_mz[b_idx] 

    # --- 3. 矩阵并行计算 ---
    
    # 计算距离矩阵 [Total_Matches, N_gt]
    dist_matrix = torch.abs(anchors - targets_set)

    # 计算 Logits
    logits = -dist_matrix / temperature

    # --- 4. 计算 Loss ---
    
    # 这里的 target_labels 对应的是 targets_set 中的列索引 (col_ind)
    # 默认 reduction='mean' 会对所有匹配项求平均
    # 原代码是 loss / batch_size (即对图求平均)，如果每个图匹配数差异巨大，
    # 建议使用 reduction='sum' 然后手动除以 batch_size
    
    loss = F.cross_entropy(logits, target_labels, reduction='sum')
    
    return loss / batch_size

def old_scalar_contrastive_loss(pred_mz, gt_mz, indices, temperature=0.01):
    """
    直接在标量空间做对比学习。
    
    pred_mz: [Batch, N_pred] (已归一化到 0-1)
    gt_mz:   [Batch, N_gt]   (已归一化到 0-1)
    indices: 匈牙利匹配结果
    temperature: 调节梯度的锐度，越小越强硬
    """
    loss = 0
    batch_size = pred_mz.shape[0]
    
    for b in range(batch_size):
        row_ind, col_ind = indices[b]
        
        # 1. 取出匹配对
        # anchor: 预测出的 m/z (只取匹配上的)
        anchor = pred_mz[b][row_ind].unsqueeze(1)  # [K, 1]
        
        # 2. 构建对比集合 (Positives + Negatives)
        # 在这个图谱中，所有的真实峰都是"候选对象"
        # 对应的 col_ind 是正样本，其他的都是负样本
        targets = gt_mz[b].unsqueeze(0)            # [1, M]
        
        # 3. 计算 L1 距离矩阵
        # dist_matrix[i, j] = |pred_i - gt_j|
        dist_matrix = torch.abs(anchor - targets)  # [K, M] Broadcast
        
        # 4. 转化为 Similarity (Sim = -Distance)
        logits = -dist_matrix / temperature
        
        # 5. 计算 Cross Entropy
        # 目标：每一行 pred_i 应该匹配到 col_ind[i] 这一列
        # label 就是 col_ind
        target_labels = torch.tensor(col_ind).to(pred_mz.device)
        
        loss += F.cross_entropy(logits, target_labels)
        
    return loss / batch_size


class SetPredictionLoss(nn.Module):
    """
    Set prediction loss for mass spectrum prediction.
    
    Uses Hungarian matching to find optimal assignment between predictions and targets,
    then computes losses for matched and unmatched predictions.
    """
    
    def __init__(
        self,
        cost_mz: float = 1.0,
        cost_intensity: float = 1.0,
        cost_confidence: float = 1.0,
        loss_mz_weight: float = 1.0,
        loss_mz_l1_weight: float = 1.0,
        loss_intensity_weight: float = 1.0,
        loss_confidence_weight: float = 1.0,
        background_confidence_weight: float = 0.1,
        temperature: float = 0.01
    ):
        """
        Initialize set prediction loss.
        
        Args:
            cost_mz: Weight for m/z cost in Hungarian matching
            cost_intensity: Weight for intensity cost in Hungarian matching
            cost_confidence: Weight for confidence cost in Hungarian matching
            loss_mz_weight: Weight for m/z contrastive loss
            loss_mz_l1_weight: Weight for m/z L1 loss
            loss_intensity_weight: Weight for intensity loss
            loss_confidence_weight: Weight for confidence loss (matched predictions)
            background_confidence_weight: Weight for background confidence loss (unmatched predictions)
            temperature: Temperature parameter for scalar contrastive loss (lower = sharper gradients)
        """
        super().__init__()
        
        self.matcher = HungarianMatcher(cost_mz, cost_intensity)
        
        self.loss_mz_weight = loss_mz_weight
        self.loss_mz_l1_weight = loss_mz_l1_weight
        self.loss_intensity_weight = loss_intensity_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.background_confidence_weight = background_confidence_weight
        self.temperature = temperature
        self.cosine_annealer = CosineAnnealer(0.01, 0.0007, 200)
    
    def forward(
        self,
        pred_mz: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_confidence_logits: torch.Tensor,
        target_mz: torch.Tensor,
        target_intensity: torch.Tensor,
        target_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute set prediction loss.
        
        Args:
            pred_mz: Predicted m/z values, shape (batch_size, num_predictions)
            pred_intensity: Predicted intensities, shape (batch_size, num_predictions)
            pred_confidence: Predicted confidences, shape (batch_size, num_predictions)
            target_mz: Target m/z values, shape (batch_size, num_targets)
            target_intensity: Target intensities, shape (batch_size, num_targets)
            target_mask: Mask for real targets, shape (batch_size, num_targets)
            
        Returns:
            Dictionary of losses
        """
        batch_size = pred_mz.size(0)
        num_predictions = pred_mz.size(1)
        
        # Perform Hungarian matching
        indices = self.matcher(
            pred_mz, pred_intensity, pred_confidence_logits.sigmoid(),
            target_mz, target_intensity, target_mask
        )
        
        # Get matched pairs
        matched_pred_mask, (batch_idx, pred_idx, target_idx) = get_matched_pairs(
            indices, batch_size, num_predictions
        )
        
        # Initialize losses
        loss_mz = torch.tensor(0.0, device=pred_mz.device)
        loss_mz_l1 = torch.tensor(0.0, device=pred_mz.device)
        loss_intensity = torch.tensor(0.0, device=pred_mz.device)
        loss_confidence_matched = torch.tensor(0.0, device=pred_mz.device)
        loss_confidence_background = torch.tensor(0.0, device=pred_mz.device)
        
        num_matched = len(pred_idx)
        
        # Compute losses for matched predictions
        if num_matched > 0:
            # Get matched predictions and targets
            matched_pred_mz = pred_mz[batch_idx, pred_idx]
            matched_pred_intensity = pred_intensity[batch_idx, pred_idx]
            # TODO: best practice is to use the sigmoid after the model
            matched_pred_confidence_logits = pred_confidence_logits[batch_idx, pred_idx]
            
            matched_target_mz = target_mz[batch_idx, target_idx]
            matched_target_intensity = target_intensity[batch_idx, target_idx]
            
            # Scalar contrastive loss for m/z
            temperature = self.cosine_annealer.get_value()
            loss_mz = new_scalar_contrastive_loss(
                pred_mz, target_mz, indices, temperature=temperature
            )
            self.cosine_annealer.step()
            
            # L1 loss for m/z
            loss_mz_l1 = F.l1_loss(matched_pred_mz, matched_target_mz)
            
            # L1 loss for intensity
            loss_intensity = F.l1_loss(matched_pred_intensity, matched_target_intensity)
            
            # Binary cross-entropy for confidence (target = 1 for matched predictions)
            target_confidence_matched = torch.ones_like(matched_pred_confidence_logits)
            loss_confidence_matched = F.binary_cross_entropy_with_logits(
                matched_pred_confidence_logits,
                target_confidence_matched
            )
        
        # Compute loss for unmatched predictions (background)
        unmatched_pred_mask = ~matched_pred_mask
        num_unmatched = unmatched_pred_mask.sum().item()
        
        if num_unmatched > 0:
            unmatched_pred_confidence_logits = pred_confidence_logits[unmatched_pred_mask]
            
            # Binary cross-entropy for confidence (target = 0 for unmatched predictions)
            target_confidence_background = torch.zeros_like(unmatched_pred_confidence_logits)
            loss_confidence_background = F.binary_cross_entropy_with_logits(
                unmatched_pred_confidence_logits,
                target_confidence_background
            )
        
        # Total loss
        total_loss = (
            self.loss_mz_weight * loss_mz +  # contrastive loss
            self.loss_mz_l1_weight * loss_mz_l1 +  # L1 loss
            self.loss_intensity_weight * loss_intensity +
            self.loss_confidence_weight * loss_confidence_matched +
            self.background_confidence_weight * loss_confidence_background
        )
        
        return {
            'loss': total_loss,
            'loss_mz': loss_mz,
            'loss_mz_l1': loss_mz_l1,
            'loss_intensity': loss_intensity,
            'loss_confidence_matched': loss_confidence_matched,
            'loss_confidence_background': loss_confidence_background,
            'num_matched': torch.tensor(num_matched, device=pred_mz.device, dtype=torch.float32),
            'num_unmatched': torch.tensor(num_unmatched, device=pred_mz.device, dtype=torch.float32),
            'matched_indices': (batch_idx, pred_idx, target_idx)
        }

