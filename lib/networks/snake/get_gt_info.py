import torch
from .utils import masks2patch
from detectron2.layers import cat
from lib.csrc.roi_align_layer.roi_align import ROIAlign
import torch
import numpy as np
import cv2
from typing import List, Tuple

class GT_infomation:
    def __init__(self,mask_size_assemble, mask_size, patch_size, scale, dct_encoding, patch_dct_encoding):
        self.mask_size_assemble = mask_size_assemble
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.scale = scale
        self.dct_encoding = dct_encoding
        self.patch_dct_encoding =  patch_dct_encoding
        #self.gt_pooler = ROIAlign # => (28,28)
        
    def crop_and_resize(self, gt_masks, rois, mask_sizes):
        #将gt_masks按预测的box裁剪后，使用roi align变为mask_sizes
        device = gt_masks.device
        batch_inds = torch.arange(len(rois), device=device).to(dtype=rois.dtype)[:, None]
        rois = torch.cat([batch_inds, rois[:,1:]], dim=1)  # Nx5
        align_class = ROIAlign((mask_sizes, mask_sizes))
        output = align_class(gt_masks[:,None,:,:], rois).squeeze(1)
        output = output >= 0.5
        return output
    
    def get_gt_mask(self,per_ins_cmask , rois):
        #针对单个实例预测 在ICD中，
        #gt_masks = []
        gt_classes = []
        gt_masks_coarse = []
        
        # for i in range(len(output['mask_preds'])):
        #     pred_masks = output['mask_preds'][i]
        #pred_masks = pred_masks[torch.arange(pred_masks.shape[0]),batch['ct_cls'][batch['ct_01'].byte()]]
        gt_masks_coarse = self.crop_and_resize(per_ins_cmask, rois, self.mask_size) # 获得对应rois的gt mask，resize为mask_size
        
        gt_masks = self.crop_and_resize(per_ins_cmask, rois, self.mask_size_assemble) #【n，mask_size_assemble，mask_size_assemble】
        gt_masks_per_image = masks2patch(gt_masks,self.scale,self.patch_size,self.mask_size_assemble) # [B*num_patch,patch_size,patch_size] num_patch即分成了多少个patch
        
        gt_masks = self.patch_dct_encoding.encode(gt_masks) # 将gt_masks encode为 dct编码的
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        
        gt_masks_coarse = self.dct_encoding.encode(gt_masks_coarse).to(dtype=torch.float32) # gt_masks_coarse也要进行encoding
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks) #gt_bfg是将dct_encoding后的bfg进行处理，0是前景 1是边缘 2是背景 

        return gt_masks,gt_masks_coarse,gt_bfg
        
        for instances_per_image in instances:

            if len(instances_per_image) == 0:
                continue

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size)
            gt_masks_coarse.append(gt_masks_per_image) #source mask concat

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size_assemble)
            # divided masks into scalexscale patch,patch size=8
            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append(gt_masks_per_image) #patch concat

            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)  # [N, dct_v_dim]
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        gt_classes = cat(gt_classes, dim=0) #[N_instanc]
        gt_masks_coarse = cat(gt_masks_coarse,dim=0)
        gt_masks_coarse = self.dct_encoding.encode(gt_masks_coarse).to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_classes,gt_masks_coarse,gt_bfg

    #TODO inference 还需要修改
    def get_gt_mask_inference(self,instances,pred_mask_logits):
        gt_masks = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if instances_per_image.has("gt_masks"):
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.pred_boxes.tensor, self.mask_size_assemble)
            else:
                #print("gt_mask is empty")
                shape = instances_per_image.pred_boxes.tensor.shape[0]
                device = instances_per_image.pred_boxes.tensor.device
                gt_masks_per_image = torch.zeros((shape,self.mask_size_assemble,self.mask_size_assemble),dtype=torch.bool).to(device)

            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append((gt_masks_per_image))

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)
        gt_masks = gt_masks.to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_bfg

    def get_gt_bfg(self, gt_masks):
        gt_bfg = gt_masks[:, 0].clone()
        gt_bfg[(gt_bfg > 0) & (gt_bfg < self.patch_size)] = 1.
        gt_bfg[gt_bfg == self.patch_size] = 2
        gt_bfg = gt_bfg.to(dtype=torch.int64)
        gt_masks = gt_masks[gt_bfg == 1, :]
        return gt_masks, gt_bfg
    
    def get_gt_classes(self, instances):
        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0)  # [N_instance]
        return gt_classes

def points_to_mask(
    points: torch.Tensor,  # (N, 2, 128), 坐标格式为 (x, y)
    image_size: Tuple[int, int],  # (H, W)
    device: str = "cpu"
) -> torch.Tensor:
    """
    将点集转换为二值掩码（多边形填充）
    Args:
        points: (N, 2, 128) - 第1维是x坐标，第2维是y坐标
        image_size: 输出掩码的尺寸 (H, W)
        device: 输出掩码的设备
    Returns:
        masks: (N, H, W) - 二值掩码（1表示多边形内，0表示背景）
    """
    N, num_points, _ = points.shape
    H, W = image_size
    masks = torch.zeros((N, H, W), dtype=torch.float32, device=device)
    
    # 转换为CPU上的numpy数组（OpenCV处理需要）
    points_np = points.cpu().numpy()
    
    for i in range(N):
        # 提取第i个实例的128个点 (2, 128) -> (128, 2)
        poly = points_np[i]  # (128, 2), 格式为 [[x0,y0], [x1,y1], ...]
        
        # 创建空白画布
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 将多边形坐标转为OpenCV所需的int32格式
        poly_int = poly.reshape(-1, 1, 2).astype(np.int32)
        
        # 填充多边形（注意OpenCV的x=列，y=行）
        cv2.fillPoly(mask, [poly_int], color=1)
        
        # 转为Tensor并存入结果
        masks[i] = torch.from_numpy(mask).to(device)
    
    return masks

def compute_mask_iou(
    pred_masks: torch.Tensor,  # (N_pred, H, W)
    gt_masks: torch.Tensor     # (N_gt, H, W)
) -> torch.Tensor:
    """
    计算预测掩码与 GT 掩码之间的 IoU 矩阵
    Returns:
        iou_matrix: (N_pred, N_gt)
    """
    # 交集 = pred AND gt
    intersection = (pred_masks[:, None] * gt_masks).sum(dim=(-2, -1))  # (N_pred, N_gt)
    
    # 并集 = pred OR gt
    union = (pred_masks[:, None] + gt_masks).clip(max=1).sum(dim=(-2, -1))  # (N_pred, N_gt)
    
    # 避免除以零
    iou_matrix = intersection / (union + 1e-6)
    return iou_matrix

def match_masks_with_classes(
    pred_masks: torch.Tensor,   # (N_pred, H, W)
    pred_classes: torch.Tensor, # (N_pred,), 预测类别索引 (0~16)
    gt_masks: torch.Tensor,     # (N_gt, H, W)
    gt_classes: torch.Tensor,   # (N_gt,), GT类别索引 (0~2)
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    匹配同类别的预测和GT掩码，允许多个预测匹配同一个GT。
    返回一个长度为N_pred的张量，每个元素是对应的GT索引（未匹配则为-1）。
    """
    iou_matrix = compute_mask_iou(pred_masks, gt_masks)  # (N_pred, N_gt)
    
    # 仅保留同类别的IoU
    class_mask = pred_classes[:, None] == gt_classes  # (N_pred, N_gt)
    iou_matrix *= class_mask.float()
    
    # 初始化结果张量（未匹配的默认为-1）
    matched_gt_indices = torch.full((pred_masks.shape[0],), -1, dtype=torch.long, device=pred_masks.device)
    
    # 遍历每个GT，允许其匹配多个预测
    for gt_idx in range(gt_masks.shape[0]):
        # 找到所有满足条件的预测（同类 + IoU ≥ 阈值）
        valid_preds = (iou_matrix[:, gt_idx] >= iou_threshold).nonzero().squeeze(-1)
        
        # 将这些预测标记为匹配当前GT
        matched_gt_indices[valid_preds] = gt_idx
    
    return matched_gt_indices  # (N_pred,)

def match_masks_multiple(
    pred_masks: torch.Tensor,   # (N_pred, H, W)
    pred_classes: torch.Tensor, # (N_pred,), 预测类别索引 (0~16)
    gt_masks: torch.Tensor,     # (N_gt, H, W)
    gt_classes: torch.Tensor,   # (N_gt,), GT类别索引 (0~2)
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    返回一个长度为 N_pred 的张量，每个元素是对应的 GT 索引（未匹配为 -1）
    """
    N_pred = pred_masks.shape[0]
    device = pred_masks.device
    
    # 初始化输出：全 -1（表示未匹配）
    pred_to_gt = torch.full((N_pred,), -1, dtype=torch.long, device=device)
    
    # 计算 IoU 矩阵并过滤不同类别
    iou_matrix = compute_mask_iou(pred_masks, gt_masks)  # (N_pred, N_gt)
    class_mask = pred_classes[:, None] == gt_classes     # (N_pred, N_gt)
    iou_matrix *= class_mask.float()
    
    # 遍历所有 GT，为每个 GT 记录所有满足条件的预测
    for gt_idx in range(gt_masks.shape[0]):
        # 找到所有同类且 IoU >= 阈值的预测
        matched_pred_mask = (iou_matrix[:, gt_idx] >= iou_threshold)
        matched_pred_indices = matched_pred_mask.nonzero().squeeze(-1)  # (K,)
        
        # 将这些预测的对应 GT 索引设为当前 gt_idx
        pred_to_gt[matched_pred_indices] = gt_idx
    
    return pred_to_gt  # (N_pred,)