import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .update import Conv2d,ConvTranspose2d
from lib.csrc.roi_align_layer.roi_align import ROIAlign
from lib.config import cfg

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
