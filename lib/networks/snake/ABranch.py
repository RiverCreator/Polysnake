import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .update import Conv2d,ConvTranspose2d
from lib.csrc.roi_align_layer.roi_align import ROIAlign
from lib.config import cfg
from .PatchDCT import PatchDCT
from .get_gt_info import GT_infomation
class AmodalBranch(nn.Module):
    def __init__(self, num_classes):
        super(AmodalBranch,self).__init__()
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

class BasicAmodalBranch(nn.Module):
    def __init__(self, num_classes):
        super(BasicAmodalBranch,self).__init__()
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
    
    def forward(self, feature, py, ct_01=None,detection=None):
        # feature : [B, 64, 168, 128]
        if detection is None:
            rois = self.get_box(py, ct_01)
        else:
            ct_01 = torch.ones([1, detection.size(0)])
            rois = self.get_box(py, ct_01.byte())
        roi_feature = self.pooler(feature, rois)
        mask_logits = self.mask_head(roi_feature)
        return mask_logits, rois