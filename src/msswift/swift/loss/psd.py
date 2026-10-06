import os
import torch
import torch.nn.functional as F

from .base import BaseLoss


# ============================================================
# 1️⃣  Localization IoU + L1 Hybrid Loss
# ============================================================

def localization_iou_l1_loss(pred_start: torch.Tensor,
                             pred_end: torch.Tensor,
                             gt_start: torch.Tensor,
                             gt_end: torch.Tensor,
                             eps: float = 1e-8) -> torch.Tensor:
    """
    Hybrid IoU + SmoothL1 loss for 1D temporal localization.
    More stable than pure IoU.

    All inputs: shape [B]
    """

    # 强制合法区间
    pred_start, pred_end = torch.minimum(pred_start, pred_end), \
                           torch.maximum(pred_start, pred_end)

    overlap = (torch.minimum(pred_end, gt_end) -
               torch.maximum(pred_start, gt_start)).clamp(min=0.)

    union = (torch.maximum(pred_end, gt_end) -
             torch.minimum(pred_start, gt_start)).clamp(min=eps)

    iou = overlap / union
    iou_loss = 1.0 - iou

    l1_loss = F.smooth_l1_loss(
        torch.stack([pred_start, pred_end], dim=-1),
        torch.stack([gt_start, gt_end], dim=-1),
        reduction='none'
    ).mean(dim=-1)

    # IoU 与 L1 各占 0.5，可调
    return (0.5 * iou_loss + 0.5 * l1_loss).mean()


# ============================================================
# 2️⃣  PSD Weighted Cross Entropy (Segment-aware)
# ============================================================

def psd_weighted_ce_loss(outputs,
                         labels,
                         loss_scale=None,
                         num_items_in_batch=None,
                         **kwargs) -> torch.Tensor:
    """
    Segment-weighted CE for:
        <think>
        Detection Result
        Localization Result

    Requires --loss_scale psd_think_det_loc
    """

    from swift.trainers import per_token_loss_func

    # per_token_loss_func 返回展平的一维向量 [B * T]，
    # 行为与上方注释版本一致：loss 本身已经对 ignore_index=-100 的 token 给出 0。
    token_loss = per_token_loss_func(outputs, labels)  # [B * T]

    # 和 per_token_loss_func 内部一致，对 labels 做 roll 后再展平，仅用于统计有效 token 数
    shifted_labels_flat = torch.roll(labels, shifts=-1, dims=-1).view(-1)
    valid_mask = (shifted_labels_flat != -100)

    if loss_scale is not None:
        # trainer 已对 loss_scale 做 roll(-1).view(-1)，与 token_loss [B*T] 对齐；此处仅 view(-1) 并送 device
        # 参考注释版本：权重本身不受 mask 影响，仅在分母处对有效位置做归一化
        loss_scale_flat = loss_scale.view(-1).to(
            device=token_loss.device,
            dtype=token_loss.dtype
        )
        weighted_sum = (token_loss * loss_scale_flat).sum()
        weight_sum = loss_scale_flat[valid_mask].sum().clamp(min=1e-8)
        return weighted_sum / weight_sum

    if num_items_in_batch is None:
        num_items_in_batch = valid_mask.sum()

    divisor = num_items_in_batch
    if isinstance(divisor, torch.Tensor):
        divisor = divisor.clamp(min=1)
    else:
        divisor = max(1, int(divisor))

    # 与注释版本保持一致：对所有 token_loss 求和，再除以有效 token 数
    return token_loss.sum() / divisor


# ============================================================
# 3️⃣  PSD Detection + Localization Joint Loss
# ============================================================
def psd_detection_localization_loss(outputs,
                                    labels,
                                    loss_scale=None,
                                    num_items_in_batch=None,
                                    trainer=None,
                                    **kwargs) -> torch.Tensor:
    """
    Final PSD Loss:

    L = Weighted CE
        + λ * Localization IoU+L1 (partially fake)
        + 0.3 * L1 penalty to zero interval (fully real)

    Expected in outputs:
        outputs["localization_pred"] -> [B, 2]
        outputs["localization_gt"]   -> [B, 2]
        outputs["detection_label"]   -> [B]

    detection_label mapping (int):
        
        0 = fully real  (penalize any non-zero predicted interval)
        1 = fully fake  (no localization loss term, only CE)
        2 = partially fake  (IoU+L1 w.r.t. ground-truth interval)
    """

    # -------------------------
    # 1️⃣ CE part
    # -------------------------
    ce_loss = psd_weighted_ce_loss(
        outputs,
        labels,
        loss_scale=loss_scale,
        num_items_in_batch=num_items_in_batch
    )

    iou_weight = float(os.environ.get('PSD_IOU_WEIGHT', '0.5'))

    if iou_weight <= 0:
        return ce_loss

    # -------------------------
    # 2️⃣ 获取 localization
    # -------------------------

    if isinstance(outputs, dict):
        pred = outputs.get('localization_pred', None)
        gt = outputs.get('localization_gt', None)
        cls_label = outputs.get('detection_label', None)
    else:
        pred = getattr(outputs, 'localization_pred', None)
        gt = getattr(outputs, 'localization_gt', None)
        cls_label = getattr(outputs, 'detection_label', None)

    if pred is None or gt is None:
        return ce_loss

    if isinstance(pred, (list, tuple)):
        pred = torch.stack(pred, dim=-1)
    if isinstance(gt, (list, tuple)):
        gt = torch.stack(gt, dim=-1)

    pred = pred.to(ce_loss.device)
    gt = gt.to(ce_loss.device)

    pred_start, pred_end = pred[..., 0], pred[..., 1]
    gt_start, gt_end = gt[..., 0], gt[..., 1]

    # -------------------------
    # 3️⃣ 仅 partially fake 计算 IoU
    # -------------------------

    if cls_label is not None:
        cls_label = cls_label.to(ce_loss.device)

        partial_mask = (cls_label == 2)   # partially fake
        real_mask = (cls_label == 0)      # fully real

        iou_loss = torch.tensor(0.0, device=ce_loss.device)

        # ---- partially fake ----
        if partial_mask.any():
            iou_loss = localization_iou_l1_loss(
                pred_start[partial_mask],
                pred_end[partial_mask],
                gt_start[partial_mask],
                gt_end[partial_mask]
            )

        # ---- fully real: 强制无区间 ----
        if real_mask.any():
            real_penalty = F.smooth_l1_loss(
                pred[real_mask],
                torch.zeros_like(pred[real_mask])
            )
            iou_loss = iou_loss + 0.3 * real_penalty

    else:
        # 没有分类标签时直接算
        iou_loss = localization_iou_l1_loss(
            pred_start,
            pred_end,
            gt_start,
            gt_end
        )

    return ce_loss + iou_weight * iou_loss

# ============================================================
# 4️⃣  BaseLoss 包装，供 loss_map 使用
# ============================================================

class PSDDetectionLocalizationLoss(BaseLoss):
    """PSD detection + localization loss 的 BaseLoss 包装。"""

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        # 避免 trainer 同时出现在 kwargs 与显参中导致 "multiple values for keyword argument 'trainer'"
        kwargs = {**kwargs, 'trainer': kwargs.get('trainer', self.trainer)}
        return psd_detection_localization_loss(
            outputs,
            labels,
            loss_scale=loss_scale,
            num_items_in_batch=num_items_in_batch,
            **kwargs
        )
