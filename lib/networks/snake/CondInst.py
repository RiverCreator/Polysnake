import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from .update import Conv2d
"""
DynamicMaskHead: 动态mask头,加载卷积权重
MaskBranch: 将坐标与特征图concat在一起
CondInst: 为整体框架,内部定义Controller和classification,然后分别调用DynamicMaskHeda和MaskBranch进行forward。
"""
class DynamicMaskHead(nn.Module):
    def __init__(self,cfg):
        super(DynamicMaskHead,self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.CONV_NUMS  #3
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS # 8
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS # mask branch的输出通道数
        self.out_feat_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE # 原condinst这里用的是FPN的P3，指的是原图到现在特征图的尺度变换比例，为4。
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS # 实验测试加上相对坐标和不加，原文中为了指定实例所在位置，因此cat了相对坐标，这里默认都是false
        
        weight_nums, bias_nums = [], [] # 记录DMH中的weight和bias的个数
        for l in range(self.num_layers): #计算weight和bias的个数
            if l == 0: #第0层时，如果cat了相对坐标，则输入维度要多两维的坐标
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1: # 最后一层则输出1维
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

        return weight_splits, bias_splits
    
    def mask_head_forward(self, features, weights, biases, num_insts):
        """
            feature: [B,C,H,W] B represent the num of instance
            weight: conv's weight, each weight represent num_insts' conv weight
            bias: same as wight
            num_insts: which represent the num of predicated instance
        """
        # 加载进Controller预测的weight和bias，
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    
class MaskBranch(nn.Module):
    def __init__(self, cfg):
        super(MaskBranch,self).__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES  # mask branch的输入channel，原文中为p3，channel数为256，这里调整为dla34的输出维度
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS # mask branch的输出channel
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS # 卷积个数，原文中是使用的4个3×3
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS # mask branch中间卷积的channel数 为128
        
        #self.refine = nn.ModuleList()# mask branch，对backbone得到的feature进行一次卷积，相当于refine
        #feature_channels = {k: v.channels for k, v in input_shape.items()} # 记录一下backbone出来的每张特征图大小

        # for in_feature in self.in_features:
        #     self.refine.append(Conv2d(
        #         feature_channels[in_feature],
        #         channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         norm=nn.BatchNorm2d(channels),
        #         activation=F.relu
        #     ))
        
        # mask branch中间卷积部分
        tower = []
        for i in range(num_convs): # 四个卷积，保持输入输出一样
            if(i==0):
                tower.append(Conv2d(
                    self.in_features,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=nn.BatchNorm2d(channels),
                    activation=F.relu
                ))
            else:
                tower.append(Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=nn.BatchNorm2d(channels),
                    activation=F.relu
                )
                )
        #最后一个卷积，控制输出channel为8
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

    def forward(self, cnn_feature):
        refine_feture = self.tower(cnn_feature)
        return refine_feture
    
class CondInst(nn.Module):
    def __init__(self, cfg):
        super(CondInst, self).__init__()
        #self.input_shape= ['fea1','fea2'] ## need modify
        self.mask_head = DynamicMaskHead(cfg)
        self.mask_branch = MaskBranch(cfg)
        self.in_channels = 128 # 为送入CenterNet的特征图的channel
        self.controller=nn.Conv2d( ## 使用3×3卷积，假设输入为(B,C,H,W),输出为(B,C1,H,W) 其中C1为要预测的参数数量，这样每个center都有一组参数。
            self.in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

    def compute_locations(self, h, w, stride, device):
        #compute 
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
    def use_gt_centers(self, output, batch):
        bacthsize, _, height, width = output['ct_hm'].size()
        wh_pred = output['wh'] ## 预测的每个点的偏移量 shape为 b 128*2 h w
        # inp_h,inp_w=batch['meta']['inp_out_hw'][:2]
        # inp_h=inp_h/snake_config.ro
        # inp_w=inp_w/snake_config.ro
        ct_01 = batch['ct_01'].byte()
        ct_ind = batch['ct_ind'][ct_01] ## 这里的ct_ind表示的是ct_heatmap中的每个点的下标，从左到右从上到下
        ct_img_idx = batch['ct_img_idx'][ct_01] ##确定图像的下标
        ct_x, ct_y = ct_ind % width, ct_ind // width  #确定对应物体中心点的坐标
        ct_img_idx = ct_img_idx % bacthsize   ## 这里是为了处理多个batch concat在一起的情况
        ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
        ct = torch.cat([ct_x, ct_y], dim=1)  ## 这里直接把ct坐标转换为（ct_num，2）shape大小的
        return ct
    
    def forward(self, output, cnn_feature, batch, detection = None):
        # cnn_feature（168,128），image（672,512）四倍
        B, N, H, W =cnn_feature.size()
        mask_feats = self.mask_branch(cnn_feature)
        stride = 4 # dla中因为只用了一个特征图 为4倍降采样
        locations = self.compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=stride, device=mask_feats.device
        )
        # instance_locations = instances.locations # 将这筛选出来的正样本的坐标偏移量
        # relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2) # (338 1 2) - (1 12880 2) 通过与所有位置的坐标相减，得到所有位置相对于该预测点的相对位置
        # relative_coords = relative_coords.permute(0, 2, 1).float() # 338 12880 2 -> 338 2 12880
        # soi = self.sizes_of_interest.float()[instances.fpn_levels] # 表示roi的size，表示这338个位置对应的fpn中的第几个特征图
        # relative_coords = relative_coords / soi.reshape(-1, 1, 1) # 将相对坐标normalize
        # relative_coords = relative_coords.to(dtype=mask_feats.dtype) # (338 2 12880)

        # mask_head_inputs = torch.cat([
        #     relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
        # ], dim=1) # 将这338个样本的位置对应的image的mask_feats拿出来 mask_feats(2,8,92,140) mask_feats[im_inds]是(338,8,92,140),reshape为（338,8,12880） 与相对坐标concat为（338,10,12880）
        if self.training or detection is None:
            ct = self.use_gt_centers(output, batch)
            ct_01 = batch['ct_01'].byte()
            im_inds = batch['ct_img_idx'][ct_01]
            im_inds = im_inds % B
        else:
            #detection = self.decode_detection(output,H, W, score_thresh = 0.03) # test val
            xs = detection[:,0][:,None]
            ys = detection[:,1][:,None]
            ct = torch.cat([xs, ys], dim=1)
            im_inds = [0] * ct.size(0) # TODO 测试的时候只能batchsize为1，可以调整一下
        mask_head_params = self.controller(cnn_feature)# output['controller']
        params = []
        mask_head_params = mask_head_params[im_inds]
        for i in range(len(mask_head_params)):
            x = ct[i][0].long()
            y = ct[i][1].long()
            params.append(mask_head_params[i,:,y,x])
        params = torch.stack(params)
        ct = ct * stride
        relative_coords = ct.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = relative_coords.to(dtype=mask_feats.dtype)
        mask_head_inputs = torch.cat([
             relative_coords, mask_feats[im_inds].reshape(ct.size(0), self.mask_branch.num_outputs, H * W)
        ], dim=1)
        
        #mask_head_params = self.controller(cnn_feature)
        # 将每个center的相对坐标concat到cnn_feature上，
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = self.mask_head.parse_dynamic_params(
            params, self.mask_head.channels,
            self.mask_head.weight_nums, self.mask_head.bias_nums
        )
        mask_logits = self.mask_head.mask_head_forward(mask_head_inputs, weights, biases, ct.size(0))
        
        return mask_logits[0]