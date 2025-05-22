import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .update import Conv2d,ConvTranspose2d
from lib.csrc.roi_align_layer.roi_align import ROIAlign
from lib.config import cfg

import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .update import Conv2d,ConvTranspose2d
from lib.csrc.roi_align_layer.roi_align import ROIAlign
from lib.config import cfg
from .PatchDCT import PatchDCT
from .get_gt_info import GT_infomation
from typing import Any, Optional, Tuple, Type
import numpy as np
import cv2

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 32, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def encode_single_point(self, x: int, y: int, image_size: tuple = (1024, 1024)):
        point = torch.tensor([[[x, y]]], dtype=torch.float32)  # (1, 1, 2)
        embedding = self.forward_with_coords(point, image_size)  # (1, 1, 128)
        return embedding.squeeze(0)  # 输出 (1, 128)
    
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding
    
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class RandomSampler(nn.Module):
    def __init__(self):
        super(RandomSampler, self).__init__()
        
        
    def mask_to_normalized(self,
        mask_coords: torch.Tensor,  # (N, 2) 或 (..., 2)，值为 [x_mask, y_mask]
        mask_size: Tuple[int, int]  # (H_mask, W_mask)，如 (28, 28)
    ) -> torch.Tensor:
        """
        将 mask 局部坐标转换为 ROI 内的归一化坐标 [0,1]。
        """
        # 归一化到 [0, 1]
        normalized = mask_coords / torch.tensor([mask_size[1] - 1, mask_size[0] - 1], 
                                            device=mask_coords.device)
        return normalized  # 形状与输入相同
    
    def normalized_to_global(self,
        normalized_coords: torch.Tensor,  # (..., 2)，范围 [0,1]
        roi_coords: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    ) -> torch.Tensor:
        """
        将归一化坐标映射回原图全局坐标。
        """
        x_min, y_min, x_max, y_max = roi_coords
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        # 全局坐标 = 左上角 + 归一化坐标 × ROI宽高
        global_coords = torch.stack([
            x_min + normalized_coords[..., 0] * roi_width,
            y_min + normalized_coords[..., 1] * roi_height
        ], dim=-1)
        
        return global_coords  # 形状与输入相同
    
    def mask_to_global(self,
        mask_coords: torch.Tensor,
        mask_size: Tuple[int, int],
        roi_coords: Tuple[float, float, float, float]
    ) -> torch.Tensor:
        # 1. 归一化到 [0,1]
        normalized = self.mask_to_normalized(mask_coords, mask_size)
        # 2. 映射到全局坐标
        global_coords = self.normalized_to_global(normalized, roi_coords)
        return global_coords
    
    def sample_point(self, error_region, roi, num_points=1):
        ""
        """
        随机采样一个点击点并返回
        """
        nonzero_coords = torch.nonzero(error_region)  # 形状 (K, 2)，K是错误像素数
        
        # 如果没有错误区域，返回空或默认值
        if len(nonzero_coords) == 0:
            return torch.zeros((num_points, 2), dtype=torch.float32)
        
        # 随机选择 num_points 个点
        rand_indices = torch.randperm(len(nonzero_coords))[:num_points]
        sampled_points = nonzero_coords[rand_indices].float()

        global_points = self.mask_to_global(sampled_points, [28, 28], roi)
        
        return global_points  # 形状 (num_points, 2)

    def sample_points_from_error(
        self,
        error_region: torch.Tensor,  # (H, W)，1表示错误区域
        num_points: int = 1
    ) -> torch.Tensor:
        """
        从错误区域随机采样坐标点（修正xy顺序）
        Returns:
            points: (num_points, 2)，格式为 (x, y)
        """
        # 获取所有错误像素的坐标（注意：nonzero返回(y,x)）
        nonzero_coords = torch.nonzero(error_region)  # (K, 2)，格式[y,x]
        
        if len(nonzero_coords) == 0:
            return torch.zeros((num_points, 2), dtype=torch.float32, device=error_region.device)
        
        # 随机选择num_points个点，并交换x,y顺序
        rand_idx = torch.randint(0, len(nonzero_coords), (num_points,))
        sampled_points = nonzero_coords[rand_idx].flip(-1).float()  # flip(-1)交换最后维度的顺序
        
        return sampled_points  # (num_points, 2)，格式[x,y]
        
    def vis_mask(self, mask):
        import numpy as np
        from PIL import Image
        
        # 假设您的掩码张量存储在变量 mask 中，形状为 (1, 1, 168, 128)
        # 这里我们使用随机数据作为示例
        #mask = np.random.randn(1, 1, 168, 128)

        # 去除多余的维度，得到形状为 (168, 128) 的二维数组
        mask = np.squeeze(mask)

        #mask_normalized = torch.sigmoid(mask)

        # 将归一化后的张量值缩放到 0-255 范围，并转换为无符号8位整数类型
        mask_scaled = (mask * 255).byte()

        # 将张量转换为 NumPy 数组
        mask_np = mask_scaled.cpu().numpy()

        # 将 NumPy 数组转换为 PIL 图像
        mask_image = Image.fromarray(mask_np)

        # 保存为 JPEG 格式的图像
        mask_image.save('vis_test.jpg', format='JPEG')
        
    def batch_sample_error_points(self,
        error_regions: torch.Tensor,  # (N, H, W)
        num_points: int = 1
    ) -> torch.Tensor:
        """
        对每个实例采样错误点
        Returns:
            all_points: (N, num_points, 2)
        """
        N = error_regions.shape[0]
        all_points = []
        
        for i in range(N):
            points = self.sample_points_from_error(error_regions[i], num_points)
            all_points.append(points)
        
        return torch.stack(all_points, dim=0)  # (N, num_points, 2)
    
    def generate_gaussian_heatmaps_points(self,
        gt_masks,
        pred_masks,
        sigma: float = 5.0,   # 高斯核标准差
    ) -> torch.Tensor:
        """
        生成高斯热度图
        Args:
            points: 采样点坐标，范围需在原图尺寸内
            image_size: 热度图大小 (H, W)
            sigma: 高斯核标准差（控制扩散范围）
        Returns:
            heatmap: (H, W) 的热度图
        """
        H, W = gt_masks.shape[-2:]
        device = pred_masks.device
        heatmap = torch.zeros((H, W), device=device)
        
        # 生成网格坐标 (H, W, 2)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        #points_list = self.sample_point_batch(gt_masks, pred_masks, rois)
        error_regions = (pred_masks != gt_masks).float()
        points_list = self.batch_sample_error_points(error_regions, 1)
        heatmaps = []
        #for x, y in points_list:
            # 计算高斯分布
        for i in range(len(points_list)):
            x, y = points_list[i][0][:]
            dist_sq = ((grid_coords - torch.tensor([x, y], device=device)) ** 2).sum(dim=-1)
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            heatmap += gaussian
            heatmaps.append(heatmap)
        # # 归一化到 [0, 1]（可选）
        if len(points_list) > 0:
            heatmap = heatmap / heatmap.max()
        
        return torch.stack(heatmaps,dim=0)
    
    def generate_gaussian_heatmaps_points_test(
        self,
        gt_masks: torch.Tensor,      # (N_gt, H, W)
        pred_masks: torch.Tensor,    # (N_pred, H, W)
        pred_to_gt: torch.Tensor,    # (N_pred,), 每个预测对应的GT索引（-1表示未匹配）
        sigma: float = 5.0,         # 高斯核标准差
    ) -> torch.Tensor:
        """
        生成高斯热度图（仅对匹配的预测-GT对计算误差区域）
        Args:
            gt_masks: 真实掩码 (N_gt, H, W)
            pred_masks: 预测掩码 (N_pred, H, W)
            pred_to_gt: 预测到GT的匹配关系 (N_pred,), -1表示未匹配
            sigma: 高斯核标准差
        Returns:
            heatmaps: List[Tensor], 每个元素是对应预测的热度图 (H, W)（未匹配的为全0）
        """
        H, W = gt_masks.shape[-2:]
        device = pred_masks.device
        heatmaps = []
        
        # 生成网格坐标 (H, W, 2)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        
        for i in range(len(pred_masks)):
            gt_idx = pred_to_gt[i].item()
            
            # 未匹配的预测生成全0热度图
            if gt_idx == -1:
                heatmaps.append(torch.zeros((H, W), device=device))
                continue
            
            # 仅计算匹配的预测-GT对的误差区域
            error_region = (pred_masks[i] != gt_masks[gt_idx]).float()
            
            # 从误差区域采样点（假设self.batch_sample_error_points返回形状为 (1, 1, 2)）
            points = self.batch_sample_error_points(error_region.unsqueeze(0), 1)
            
            # 如果没有误差点，生成全0热度图
            if len(points) == 0 or len(points[0]) == 0:
                heatmaps.append(torch.zeros((H, W), device=device))
                continue
            
            # 生成高斯热度图
            heatmap = torch.zeros((H, W), device=device)
            for point in points[0]:  # 遍历所有采样点
                x, y = point[0], point[1]
                dist_sq = ((grid_coords - torch.tensor([x, y], device=device)) ** 2).sum(dim=-1)
                gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
                heatmap += gaussian
            
            # 归一化（可选）
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmaps.append(heatmap)
        
        return torch.stack(heatmaps,dim=0)
    
    def random_scale_boxes(self,
        boxes: torch.Tensor,
        width_scale_range: Tuple[float, float] = (0.8, 1.5),  # 宽度缩放范围
        height_scale_range: Tuple[float, float] = (0.8, 1.5),  # 高度缩放范围
    ) -> torch.Tensor:
        """
        对 Box 的宽度和高度分别进行随机缩放,random_scale_boxes->generate_guassian_heatmaps
        Args:
            boxes: (N, 4)，格式为 [x_min, y_min, x_max, y_max]
            width_scale_range: 宽度缩放因子的范围 (min, max)
            height_scale_range: 高度缩放因子的范围 (min, max)
        Returns:
            scaled_boxes: (N, 4)
        """
        # 计算中心点和当前宽高
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2  # (N, 2)
        widths = boxes[:, 2] - boxes[:, 0]           # (N,)
        heights = boxes[:, 3] - boxes[:, 1]          # (N,)
        
        # 独立生成宽度和高度的随机缩放因子
        width_scale = torch.empty(len(boxes)).uniform_(*width_scale_range).to(boxes.device)  # (N,)
        height_scale = torch.empty(len(boxes)).uniform_(*height_scale_range).to(boxes.device)  # (N,)
        
        # 缩放宽高（可能扩张或缩小）
        new_widths = widths * width_scale    # (N,)
        new_heights = heights * height_scale # (N,)
        
        # 计算缩放后的边界框
        x_min = centers[:, 0] - new_widths / 2
        y_min = centers[:, 1] - new_heights / 2
        x_max = centers[:, 0] + new_widths / 2
        y_max = centers[:, 1] + new_heights / 2
        
        scaled_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        return scaled_boxes
    
    
    def generate_smooth_rectangular_heatmaps(self,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
        decay_ratio: float = 0.1,  # 边缘衰减比例（相对于框宽度/高度）
    ) -> torch.Tensor:
        H, W = image_size
        N = boxes.shape[0]
        device = boxes.device
        heatmaps = torch.zeros((N, H, W), device=device)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        for i in range(N):
            x_min, y_min, x_max, y_max = boxes[i]
            width = x_max - x_min
            height = y_max - y_min
            
            # 计算水平方向的衰减（左右边缘）
            x_decay = decay_ratio * width
            x_left = torch.clamp((x_grid - x_min) / x_decay, 0, 1)
            x_right = torch.clamp((x_max - x_grid) / x_decay, 0, 1)
            x_weight = torch.min(x_left, x_right)
            
            # 计算垂直方向的衰减（上下边缘）
            y_decay = decay_ratio * height
            y_top = torch.clamp((y_grid - y_min) / y_decay, 0, 1)
            y_bottom = torch.clamp((y_max - y_grid) / y_decay, 0, 1)
            y_weight = torch.min(y_top, y_bottom)
            
            # 组合权重（框内为1，边缘平滑衰减）
            heatmaps[i] = torch.min(x_weight, y_weight)
        
        return heatmaps  # (N, H, W)
    def generate_smooth_rectangular_heatmaps_test(
        self,
        gt_boxes: torch.Tensor,      # (N_gt, 4) 真实框 [x_min, y_min, x_max, y_max]
        pred_to_gt: torch.Tensor,    # (N_pred,) 每个预测对应的GT索引（-1表示未匹配）
        image_size: Tuple[int, int],
        decay_ratio: float = 0.1,    # 边缘衰减比例
    ) -> torch.Tensor:
        """
        根据预测与真实框的对应关系生成平滑矩形热度图
        Args:
            gt_boxes: 真实框坐标 (N_gt, 4)
            pred_to_gt: 预测到真实框的匹配关系 (N_pred,)
            image_size: 输出热度图尺寸 (H, W)
            decay_ratio: 边缘衰减比例
        Returns:
            heatmaps: (N_pred, H, W) 每个预测对应的热度图（未匹配的为全0）
        """
        H, W = image_size
        N_pred = len(pred_to_gt)
        device = gt_boxes.device
        
        # 初始化输出（全0，未匹配的预测会保持为0）
        heatmaps = torch.zeros((N_pred, H, W), device=device)
        
        # 生成网格坐标 (H, W, 2)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        for pred_idx in range(N_pred):
            gt_idx = pred_to_gt[pred_idx].item()
            
            # 未匹配的预测跳过（保持全0）
            if gt_idx == -1:
                continue
                
            # 获取对应的真实框坐标
            x_min, y_min, x_max, y_max = gt_boxes[gt_idx]
            width = x_max - x_min
            height = y_max - y_min
            
            # 计算水平方向的衰减（左右边缘）
            x_decay = decay_ratio * width
            x_left = torch.clamp((x_grid - x_min) / x_decay, 0, 1)
            x_right = torch.clamp((x_max - x_grid) / x_decay, 0, 1)
            x_weight = torch.min(x_left, x_right)
            
            # 计算垂直方向的衰减（上下边缘）
            y_decay = decay_ratio * height
            y_top = torch.clamp((y_grid - y_min) / y_decay, 0, 1)
            y_bottom = torch.clamp((y_max - y_grid) / y_decay, 0, 1)
            y_weight = torch.min(y_top, y_bottom)
            
            # 组合权重（框内为1，边缘平滑衰减）
            heatmaps[pred_idx] = torch.min(x_weight, y_weight)
        
        return heatmaps
    def generate_gaussian_heatmaps_box(self,
        boxes: torch.Tensor,          # (N, 4)
        image_size: Tuple[int, int],  # (H, W)
        sigma_ratio: float = 0.1      # 高斯核标准差与 Box 最大边长的比例
    ) -> torch.Tensor:
        H, W = image_size
        N = len(boxes)
        device = boxes.device
        heatmaps = torch.zeros((N, H, W), device=device)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        
        for i in range(N):
            x_min, y_min, x_max, y_max = boxes[i]
            center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2], device=device)
            
            # 动态计算 sigma（基于缩放后的 Box 大小）
            box_width = x_max - x_min
            box_height = y_max - y_min
            sigma = sigma_ratio * max(box_width, box_height)
            
            dist_sq = ((grid_coords - center) ** 2).sum(dim=-1)
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            heatmaps[i] = gaussian
        
        # 归一化到 [0, 1]
        heatmaps = heatmaps / heatmaps.max(dim=(1, 2), keepdim=True).values
        return heatmaps
    
class DCTMaskBranch(nn.Module):
    def __init__(self, num_classes):
        super(DCTMaskBranch,self).__init__()
        #self.pooler = ROIAlign((cfg.roi_h, cfg.roi_w))
        
        #self.pooler = ROIAlign(((int)cfg.roi_h/2, (int)cfg.roi_w/2)) # => (28,28) 为了适配PatchDCT 更改为（14,14）
        self.pooler = ROIAlign((14,14)) # => (28,28) 为了适配PatchDCT 更改为（14,14）
        self.fine_mask_pooler = ROIAlign((42,42))  # （42,42）
        self.conv1 = Conv2d(
                64,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.conv2 = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.downsample = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.deconv = ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.predicator =  Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.mask_head = nn.Sequential(
            self.conv1,
            self.conv2,
            self.downsample,
            self.deconv,
            self.predicator
        ) #TODO 设计maks head 输入feature map为（B，64, 168，128）可以设计为两种，类相关和类不相关的
        self.dct_mask_head = PatchDCT(input_shape=(64,14,14),num_classes=num_classes,conv_dims=[64,64,64,64,64],
                                      dct_vector_dim=300, mask_size=112, hidden_features=1024, fine_features_resolution=42,
                                      mask_size_assemble=112, patch_size=8, patch_dct_vector_dim=6, mask_loss_para=1.0,
                                      dct_loss_type = 'l1', patch_threshold=0.3, eval_gt=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def get_box(self, py, ct_01):
        xmax, _ = torch.max(py[:,:,0], dim = 1)
        xmin, _ = torch.min(py[:,:,0], dim = 1)
        ymax, _ = torch.max(py[:,:,1], dim = 1)
        ymin, _ = torch.min(py[:,:,1], dim = 1)
        box_roi = torch.cat([xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]],dim = 1) 
        ind = torch.cat([torch.full([ct_01[i].sum()], i) for i in range(len(ct_01))], dim=0)
        ind = ind.to(box_roi.device).float()
        roi = torch.cat([ind[:, None], box_roi], dim=1)
        return roi
    
    def forward(self, feature, fine_feature, py, batch=None,detection=None):
        # feature : [B, 64, 168, 128]
        if batch is not None:
            rois = self.get_box(py, batch['ct_01'].byte())
            roi_feature = self.pooler(feature, rois)
        else:
            ct_01 = torch.ones([1, detection.size(0)])
            rois = self.get_box(py, ct_01.byte())
        roi_feature = self.pooler(feature, rois)
        fine_mask_feature = self.fine_mask_pooler(fine_feature, rois)
        dct_mask_logits, bfg, patch_vectors, mask_logits = self.dct_mask_head(roi_feature, fine_mask_feature, batch=batch, detection=detection)
        #mask_logits = self.mask_head(roi_feature)
        
        return mask_logits, rois, dct_mask_logits, bfg, patch_vectors

class AmodalBranch(nn.Module):
    def __init__(self, num_classes):
        super(AmodalBranch,self).__init__()
        self.pooler = ROIAlign((cfg.roi_h, cfg.roi_w)) # => (28,28)
        self.conv1 = Conv2d(
                64,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.conv2 = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.downsample = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.deconv = ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.predicator =  Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.mask_head = nn.Sequential(
            self.conv1,
            self.conv2,
            self.downsample,
            self.deconv,
            self.predicator
        ) #TODO 设计maks head 输入feature map为（B，64, 168，128）可以设计为两种，类相关和类不相关的

    def get_box(self, py, ct_01):
        xmax, _ = torch.max(py[:,:,0], dim = 1)
        xmin, _ = torch.min(py[:,:,0], dim = 1)
        ymax, _ = torch.max(py[:,:,1], dim = 1)
        ymin, _ = torch.min(py[:,:,1], dim = 1)
        box_roi = torch.cat([xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]],dim = 1) 
        ind = torch.cat([torch.full([ct_01[i].sum()], i) for i in range(len(ct_01))], dim=0)
        ind = ind.to(box_roi.device).float()
        roi = torch.cat([ind[:, None], box_roi], dim=1)
        return roi
    
    def forward(self, feature, py, ct_01):
        # feature : [B, 64, 168, 128]
        rois = self.get_box(py, ct_01)
        roi_feature = self.pooler(feature, rois)
        mask_logits = self.mask_head(roi_feature)
        return mask_logits, rois


class VisibleAmodalBranch(nn.Module):
    def __init__(self, num_classes):
        super(VisibleAmodalBranch,self).__init__()
        self.pooler = ROIAlign((cfg.roi_h, cfg.roi_w)) # => (28,28)
        self.conv1 = Conv2d(
                64,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.conv2 = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.downsample = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.deconv = ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.predicator =  Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.mask_head = nn.Sequential(
            self.conv1,
            self.conv2,
            self.downsample,
            self.deconv,
            self.predicator
        ) #TODO 设计maks head 输入feature map为（B，64, 168，128）可以设计为两种，类相关和类不相关的

        self.vis_conv1 = Conv2d(
                64,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.vis_conv2 = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.vis_downsample = Conv2d(
                256,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=nn.BatchNorm2d(256),
                activation=F.relu,
        )
        self.vis_deconv = ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.vis_predicator =  Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.vis_mask_head = nn.Sequential(
            self.vis_conv1,
            self.vis_conv2,
            self.vis_downsample,
            self.vis_deconv,
            self.vis_predicator
        ) #TODO 设计maks head 输入feature map为（B，64, 168，128）可以设计为两种，类相关和类不相关的        
        
        self.trans_conv = Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    def get_box(self, py, ct_01):
        xmax, _ = torch.max(py[:,:,0], dim = 1)
        xmin, _ = torch.min(py[:,:,0], dim = 1)
        ymax, _ = torch.max(py[:,:,1], dim = 1)
        ymin, _ = torch.min(py[:,:,1], dim = 1)
        box_roi = torch.cat([xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]],dim = 1) 
        ind = torch.cat([torch.full([ct_01[i].sum()], i) for i in range(len(ct_01))], dim=0)
        ind = ind.to(box_roi.device).float()
        roi = torch.cat([ind[:, None], box_roi], dim=1)
        return roi
    
    def forward(self, feature, py, ct_01):
        # feature : [B, 64, 168, 128]
        rois = self.get_box(py, ct_01)
        roi_feature = self.pooler(feature, rois)
        mask_logits = self.mask_head(roi_feature)
        vis_roi_feature = self.trans_conv(roi_feature)
        vis_mask_logits = self.vis_mask_head(vis_roi_feature)
        return mask_logits, vis_mask_logits, rois