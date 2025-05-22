from matplotlib.pyplot import box
from .snake import Snake,GAT
from .update import BasicUpdateBlock
from lib.utils import data_utils
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .ABranch import AmodalBranch,DCTMaskBranch,PositionEmbeddingRandom,RandomSampler
from . import get_gt_info 
from typing import Tuple
import cv2
class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.iter = cfg.iter_num  # iteration number
        self.score_thresh=cfg.score_thresh
        #这里的state_dim表示128个点的特征向量
        #self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2 + 1, conv_type='dgrid', need_fea=True) #即文章中用来进行特征聚合，然后输出g_{k-1}的模块
        if(cfg.use_box):
            if(cfg.use_interactive):
                self.evolve_gcn = GAT(in_features=64 + 2 + 1 + 2, n_hidden= 256, out_features= 64, n_heads=2,concat=True, use_gat = cfg.use_gat)
            else:
                self.evolve_gcn = GAT(in_features=64 + 2 + 1, n_hidden= 256, out_features= 64, n_heads=4,concat=True, use_gat = cfg.use_gat)
        else:
            self.evolve_gcn = GAT(in_features=64 + 2, n_hidden= 256, out_features= 64, n_heads=2,concat=True, use_gat = cfg.use_gat)
        self.update_block = BasicUpdateBlock() ## 即文章中使用gru的模块
        if(cfg.use_dct):
            self.box_mask_head = DCTMaskBranch(cfg.num_classes)
            self.vis_mask_head = DCTMaskBranch(cfg.num_classes)
        else:
            self.box_mask_head = AmodalBranch(cfg.num_classes)
            self.vis_mask_head = AmodalBranch(cfg.num_classes)
        if(cfg.use_interactive):
            self.iterative_sampler = RandomSampler()
            self.pe_random = PositionEmbeddingRandom()
        #self.classify_block= ClassifyBlock(1024, cfg.num_classes)
        #self.mcr=Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', need_fea=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch) # init中存放gt和ct对应的在batch中的图片编号
        output.update({'i_gt_py': init['i_gt_py'], 'per_ins_cmask': init['per_ins_cmask'], 'per_vis_cmask': init['per_vis_cmask']}) # 将gt加到output中保存
        return init

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
        
    def visualize_instance_points(self,
        points: torch.Tensor,  # (N, 128, 2) 轮廓点数据
        index: int,            # 要可视化的实例索引
        image_size: Tuple[int, int] = (168, 128),  # 图像尺寸 (H, W)
        save_path: str = "vis_point.jpg",  # 保存路径
        point_color: Tuple[int, int, int] = (0, 0, 255),  # 点颜色 (BGR)
        line_color: Tuple[int, int, int] = (0, 255, 0),   # 连线颜色
        point_radius: int = 3,  # 点半径
        line_thickness: int = 2  # 连线粗细
    ):
        """
        可视化指定索引的实例轮廓点
        """
        # 检查索引是否有效
        if index < 0 or index >= points.shape[0]:
            raise ValueError(f"Index {index} out of range [0, {points.shape[0]-1}]")
        
        # 创建空白图像 (H, W, 3)
        H, W = image_size
        image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 获取指定实例的点并转为numpy数组
        instance_points = points[index].cpu().numpy()  # (128, 2)
        
        # 绘制连线 (闭合多边形)
        pts = instance_points.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=line_color, thickness=line_thickness)
        
        # 绘制每个点
        for (x, y) in instance_points:
            cv2.circle(image, (int(x), int(y)), point_radius, point_color, -1)
        
        # 保存图像
        cv2.imwrite(save_path, image)
        #print(f"Visualization saved to {save_path}")
    
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, box_pred=None, vis_pred=None, simulate_points = None, simulate_boxes = None, type="training"):  # i_it_poly为init point，c_it_poly为相对init point，ind为标注ct为batch中哪个图片的
        if len(i_it_poly) == 0:
            return torch.empty(0, 128, 2)
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)  ## cnn_featuer为b c h w  i_it_poly为(n,128,2),n为center个数
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  ### 将坐标对应的feature进行采样，每个点对应的feature即为长度为c的向量，故init_feature大小为n c 128，n为center个数，c为cnn_feature的channel
        x_min = torch.min(i_it_poly[..., 0], dim=-1)[0]
        y_min = torch.min(i_it_poly[..., 1], dim=-1)[0]
        x_max = torch.max(i_it_poly[..., 0], dim=-1)[0]
        y_max = torch.max(i_it_poly[..., 1], dim=-1)[0]
        ins_h = y_max - y_min
        ins_w = x_max - x_min
        if(cfg.use_interactive):
            point_prompt = snake_gcn_utils.get_mask_probility(simulate_points.to(i_it_poly.device), i_it_poly)
            box_prompt = snake_gcn_utils.get_mask_probility(simulate_boxes.to(i_it_poly.device), i_it_poly)
        if(cfg.use_dct):
            tx = (c_it_poly[..., 0]/ins_w[:, None]* cfg.roi_w* 4)
            ty = (c_it_poly[..., 1]/ins_h[:, None]* cfg.roi_h* 4)
        else:
            tx = (c_it_poly[..., 0]/ins_w[:, None]* cfg.roi_w)
            ty = (c_it_poly[..., 1]/ins_h[:, None]* cfg.roi_h)
        relative_box_poly = torch.stack((tx,ty),dim=2)
        if(cfg.use_box):
            probs = snake_gcn_utils.get_mask_probility(box_pred, relative_box_poly)
            if(type == "training"):
                vis_probs = snake_gcn_utils.get_mask_probility(vis_pred, i_it_poly)
            else:
                vis_probs = snake_gcn_utils.get_mask_probility(vis_pred, relative_box_poly)
            init_feature = init_feature * (1 + 0.1* vis_probs.sigmoid())
            if(cfg.use_interactive):
                init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1), probs.sigmoid(), point_prompt, box_prompt], dim=1)
            else:
                init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1), probs.sigmoid()], dim=1)  ## 论文中提到的将相对坐标信息与之concat，提供一个相对坐标信息 c_it_poly为（n，128，2），为了能够匹配上将其转换为n 2 128 这样最终feature 大小为n c+2 128
        else:
            init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)  ## snake中进行信息聚合 并预测偏移，对应于文章中的feature aggregation模块
        return i_poly_fea

    def get_attn_score(self, conv_block, attention_feature, i_it_poly,c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = attention_feature.size(2), attention_feature.size(3)  ## cnn_featuer为b c h w  i_it_poly为(n,128,2),n为center个数
        attn_score = snake_gcn_utils.get_gcn_feature(attention_feature, i_it_poly, ind, h, w)  ### 将坐标对应的feature进行采样，每个点对应的feature即为长度为c的向量，故init_feature大小为n c 128，n为center个数，c为cnn_feature的channel
        attn_score = torch.cat([attn_score, c_it_poly.permute(0, 2, 1)], dim=1)
        attn_score = conv_block(attn_score)
        return attn_score.squeeze(1).unsqueeze(2)
    
    def use_gt_detection(self, output, batch):
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

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1,  2)  
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)  ## 这里直接取gt点位置的偏移量

        ct_x, ct_y = ct_x[:, None].float(), ct_y[:, None].float()
        ct = torch.cat([ct_x, ct_y], dim=1)  ## 这里直接把ct坐标转换为（ct_num，2）shape大小的
        # ct[:,0]=ct[:,0]/inp_w[0]
        # ct[:,1]=ct[:,1]/inp_h[0]
        #### ct_offset这里假定输出的为归一化后的偏移量，因此需要乘上w和h
        # ct_offset[:,:,0]*w ct_offset[:,:,1]*=h
        # ct_offset[:,:,0]=ct_offset[:,:,0]*inp_h[0]
        # ct_offset[:,:,1]=ct_offset[:,:,1]*inp_w[1]
        init_polys = ct_offset + ct.unsqueeze(1).expand(ct_offset.size(0), ct_offset.size(1), ct_offset.size(2)) #将offset加到对应的ct坐标上
        
        output.update({'poly_init': init_polys * snake_config.ro})  ## 对应到原图尺寸大小的初始点
        return init_polys

    def clip_to_image(self, poly, h, w):
        poly[..., :2] = torch.clamp(poly[..., :2], min=0)
        poly[..., 0] = torch.clamp(poly[..., 0], max=w - 1)
        poly[..., 1] = torch.clamp(poly[..., 1], max=h - 1)
        return poly

    def decode_detection(self, output, h, w, score_thresh = 0.03):
        ct_hm = output['ct_hm']
        wh = output['wh']
        #detection = torch.cat([ct, scores, clses], dim=2) ct 占(1,1000,2)表示中心点位置（像素位置），其余两个占(1,1000,1)
        poly_init, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh, K=1000)

        valid = detection[0, :, 2] >= score_thresh  # min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = self.clip_to_image(poly_init, h, w)
        output.update({'poly_init_infer': init_polys * snake_config.ro, 'detection': detection})
        return poly_init, detection
    
    def postprocess(self, pred_masks, output_height, output_width):
        for i in range(len(pred_masks)):
            # pred_masks[i] = pred_masks[i].expand(1,-1,-1,-1)
            pred_masks[i] = F.interpolate(pred_masks[i],size=(output_height, output_width),mode="bilinear", align_corners=False)
        return pred_masks
    def get_box(self, py):
        xmax, _ = torch.max(py[:,:,0], dim = 1)
        xmin, _ = torch.min(py[:,:,0], dim = 1)
        ymax, _ = torch.max(py[:,:,1], dim = 1)
        ymin, _ = torch.min(py[:,:,1], dim = 1)
        box_roi = torch.cat([xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]],dim = 1)
        return box_roi
        
    def forward_with_dct(self, output, cnn_feature, fine_feature, batch, more_info = None):
        box_mask_preds = []
        vis_mask_preds = []
        rois = [] 
        vis_dct_pred_x = []
        vis_dct_pred_bfg = []
        vis_dct_pred_patch_vector = []
        amodal_dct_pred_x = []
        amodal_dct_pred_bfg = []
        amodal_dct_pred_patch_vector = []
        point_guassian_heatmaps = []
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)  # init中为gt和py_ind(标记ct属于batch中的第几张图片)，output中也加入了gt的信息
            #### modify: poly init为是对应于四倍降采样特征图上
            # 
            # 的坐标的，这里全部预测的坐标和偏移量都假定归一化了，因此都需要乘上对应的w和h
            poly_init = self.use_gt_detection(output, batch) #训练的时候，这里直接使用的gt center来对点进行初始化 ，获得dla模块中推理得到的偏移量
            poly_init = poly_init.detach()
            #get_gt_info.match_masks_multiple(pred_masks.to("cuda"),detection[:,3],gt_masks.to("cuda"),more_info['ct_cls'].squeeze(0).to("cuda"))
            #gt_masks=get_gt_info.points_to_mask(more_info['i_gt_py'].squeeze(0),(cnn_feature.shape[2],cnn_feature.shape[3]))
            #pred_masks=get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3]))
            if cfg.use_box:
                vis_mask_pred, _, vis_dct_mask_logits, vis_bfg, vis_patch_vectors = self.vis_mask_head(cnn_feature, fine_feature, poly_init, batch=batch)
                box_mask_pred, roi, amodal_dct_mask_logits, amodal_bfg, amodal_patch_vectors = self.box_mask_head(cnn_feature, fine_feature, poly_init, batch=batch)
                
                box_mask_preds.append(box_mask_pred)
                vis_mask_preds.append(vis_mask_pred)
                rois.append(roi)
                amodal_dct_pred_x.append(amodal_dct_mask_logits)
                amodal_dct_pred_bfg.append(amodal_bfg)
                amodal_dct_pred_patch_vector.append(amodal_patch_vectors)
                vis_dct_pred_x.append(vis_dct_mask_logits)
                vis_dct_pred_bfg.append(vis_bfg)
                vis_dct_pred_patch_vector.append(vis_patch_vectors)
                
            py_pred = poly_init * snake_config.ro  #乘了个4，对应到原图的尺寸，而他这里使用的feature map是经过4倍降采样的
            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init) #将坐标转换为相对于最左以及最上的相对坐标
            #i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],vis_mask_preds[-1][torch.arange(vis_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])  # n*64*128
                
            if cfg.use_box:
                if(cfg.use_interactive):
                    simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points(init['per_ins_cmask'],get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3])).to(init['per_ins_cmask'].device))
                    point_guassian_heatmaps.append(simulate_points_heatmaps)
                    gt_boxes = self.get_box(batch['i_gt_py'][torch.arange(batch['i_gt_py'].shape[0])][batch['ct_01'].byte()])
                    scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                    simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps(scaled_boxes,(cnn_feature.shape[2],cnn_feature.shape[3]))
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1],init['per_vis_cmask'].unsqueeze(1),simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1],init['per_vis_cmask'].unsqueeze(1))
            else:
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'])
            net = torch.tanh(i_poly_fea)  ## 初始h0，就是feature aggregation得到的mid feature经过一个tanh计算
            i_poly_fea = F.leaky_relu(i_poly_fea)
            py_preds = []
            cls_scores= []
            for i in range(self.iter):  ## 因为初始点需要单独通过中心点来获得，因此先进行处理后，再进行迭代 ####不过他这里代码执行还是总共只执行了self.iter次迭代，因为他这里是在循环开头用gru计算偏移量的
                net, offset = self.update_block(net, i_poly_fea) # gru模块，输出net(论文中的hk)和偏移量  net送入下一轮迭代中
                #cls_score= self.classify_block(net)
                #cls_scores.append(cls_score)
                #### offset
                # offset[:,:,0]*inp_w offset[:,:,1]*=inp_h
                py_pred = py_pred + snake_config.ro * offset# * attn_score
                py_preds.append(py_pred)

                py_pred_sm = py_pred / snake_config.ro
                if cfg.use_box:
                    box_mask_pred, roi, amodal_dct_mask_logits, amodal_bfg, amodal_patch_vectors = self.box_mask_head(cnn_feature, fine_feature, py_pred_sm, batch=batch)
                    vis_mask_pred, _, vis_dct_mask_logits, vis_bfg, vis_patch_vectors = self.vis_mask_head(cnn_feature, fine_feature, py_pred_sm, batch=batch)
                    
                    box_mask_preds.append(box_mask_pred)
                    vis_mask_preds.append(vis_mask_pred)
                    rois.append(roi)
                    amodal_dct_pred_x.append(amodal_dct_mask_logits)
                    amodal_dct_pred_bfg.append(amodal_bfg)
                    amodal_dct_pred_patch_vector.append(amodal_patch_vectors)
                    vis_dct_pred_x.append(vis_dct_mask_logits)
                    vis_dct_pred_bfg.append(vis_bfg)
                    vis_dct_pred_patch_vector.append(vis_patch_vectors)

                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                
                #i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],vis_mask_preds[-1][torch.arange(vis_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])
                if cfg.use_box:
                    if(cfg.use_interactive):
                        simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points(init['per_ins_cmask'],get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3])).to(init['per_ins_cmask'].device))
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1],init['per_vis_cmask'].unsqueeze(1),simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                    else:
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1],init['per_vis_cmask'].unsqueeze(1))
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
                #attn_score = self.get_attn_score(self.boundary_coefficient, attention_feature, py_pred_sm, c_py_pred, init['py_ind'])
                i_poly_fea = F.leaky_relu(i_poly_fea)
            # ret.update({'py_pred': py_preds, 'vis_py_pred': vis_py_preds,'i_gt_py': output['i_gt_py'] * snake_config.ro,'vis_i_gt_py': output['vis_i_gt_py'] * snake_config.ro, 'mask_preds': box_mask_preds, 'vis_mask_preds' : vis_mask_preds,'rois': rois,'vis_rois': vis_rois,
            #             'dct_x': dct_stage_pred_x, 'dct_bfg':dct_stage_pred_bfg, 'dct_patch_vector':dct_stage_pred_patch_vector, 'ind':init['py_ind']})
            ret.update({'vis_dct_x': vis_dct_pred_x, 'vis_dct_bfg':vis_dct_pred_bfg, 'vis_dct_patch_vector':vis_dct_pred_patch_vector,
                        'amodal_dct_x': amodal_dct_pred_x, 'amodal_dct_bfg':amodal_dct_pred_bfg, 'amodal_dct_patch_vector':amodal_dct_pred_patch_vector})
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro, 'cls_scores': cls_scores, 'mask_preds': box_mask_preds,'vis_mask_preds': vis_mask_preds,'rois': rois})

        if not self.training:
            with torch.no_grad():
                test_box_mask_preds = []
                test_vis_mask_preds = []
                init = self.prepare_training(output, more_info)
                py_preds = []
                poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3),self.score_thresh)
                device = detection.device
                # poly_init_loss = self.use_gt_detection(output, batch)
                # init = snake_gcn_utils.prepare_training(output, batch) # init中存放gt和ct对应的在batch中的图片编号
                # ret.update({'i_gt_py': init['i_gt_py']* snake_config.ro}) # 将gt加到output中保存
                ind = torch.zeros((poly_init.size(0)))
                py_preds.append(poly_init)
                py_pred = poly_init * snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init)
                ct_01 = torch.ones([1, detection.size(0)])
                
                if cfg.use_box:
                    box_mask_pred, roi, _, _, _  = self.box_mask_head(cnn_feature, fine_feature, poly_init, detection=detection)
                    vis_mask_pred, roi, _, _, _  = self.vis_mask_head(cnn_feature, fine_feature, poly_init, detection=detection)
                    test_box_mask_preds.append(box_mask_pred)
                    test_vis_mask_preds.append(vis_mask_pred)
                    if(cfg.use_interactive):
                        pred_masks=get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3]),device)
                        pred_to_gt = get_gt_info.match_masks_multiple(pred_masks.to(device), detection[:,3], more_info['per_ins_cmask'].squeeze(0).to(device),more_info['ct_cls'].squeeze(0).to(device))
                        simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points_test(more_info['per_ins_cmask'].squeeze(0).to(device),pred_masks, pred_to_gt)
                        gt_boxes = self.get_box(more_info['i_gt_py'][torch.arange(more_info['i_gt_py'].shape[0])][more_info['ct_01'].byte()])
                        scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                        simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps_test(scaled_boxes,pred_to_gt,(cnn_feature.shape[2],cnn_feature.shape[3]))
                
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                                ind, test_box_mask_preds[-1],test_vis_mask_preds[-1],simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                    else:
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                                ind, test_box_mask_preds[-1],test_vis_mask_preds[-1])
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                              ind)
                #py_preds=[]
                if len(py_pred) != 0:
                    net = torch.tanh(i_poly_fea)
                    i_poly_fea = F.leaky_relu(i_poly_fea)
                    for i in range(self.iter):
                        net, offset = self.update_block(net, i_poly_fea)
                        
                        py_pred = py_pred + snake_config.ro * offset
                        py_pred_sm = py_pred / snake_config.ro
                        py_preds.append(py_pred_sm)
                        if cfg.use_box:
                            box_mask_pred, roi, _, _, _  = self.box_mask_head(cnn_feature, fine_feature, py_pred_sm, detection=detection)
                            vis_mask_pred, _, _, _, _  = self.vis_mask_head(cnn_feature, fine_feature, py_pred_sm, detection=detection)
                            test_box_mask_preds.append(box_mask_pred)
                            test_vis_mask_preds.append(vis_mask_pred)
                        if i != (self.iter - 1):                     
                            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                            if cfg.use_box:
                                if(cfg.use_interacive):
                                    pred_masks=get_gt_info.points_to_mask(py_pred_sm,(cnn_feature.shape[2],cnn_feature.shape[3]),device)
                                    pred_to_gt = get_gt_info.match_masks_multiple(pred_masks.to(device), detection[:,3], more_info['per_ins_cmask'].squeeze(0).to(device),more_info['ct_cls'].squeeze(0).to(device))
                                    simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points_test(more_info['per_ins_cmask'].squeeze(0).to(device),pred_masks, pred_to_gt)
                                    scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                                    simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps_test(scaled_boxes,pred_to_gt,(cnn_feature.shape[2],cnn_feature.shape[3]))
                                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind, test_box_mask_preds[-1],test_vis_mask_preds[-1],simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1)) #init['ind'])
                                else:
                                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind, test_box_mask_preds[-1],test_vis_mask_preds[-1]) #init['ind'])
                            else:
                                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind) #init['ind'])
                            i_poly_fea = F.leaky_relu(i_poly_fea)

                    final_py_preds = [py_pred_sm]
                    py_preds.append(py_pred_sm)
                    ret.update({'py': py_preds})
                    #ret.update({'amodal_preds': test_box_mask_preds,'vis_mask_preds': test_vis_mask_preds})
                else:
                    final_py_preds = [i_poly_fea]
                    ret.update({'py': final_py_preds})
        return output
    
    def forward_with_box(self, output, cnn_feature, batch, more_info = None):
        #boundary_score=output['mask'].sigmoid()
        #attention_feature = self.occlusionatte(1-boundary_score)
        box_mask_preds = []
        vis_mask_preds = []
        point_guassian_heatmaps = []
        rois = [] 
        ret = output
        #inp_h,inp_w=batch['meta']['inp_out_hw'][:2]
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)  # init中为gt和py_ind(标记ct属于batch中的第几张图片)，output中也加入了gt的信息
            #### modify: poly init为是对应于四倍降采样特征图上
            # 
            # 的坐标的，这里全部预测的坐标和偏移量都假定归一化了，因此都需要乘上对应的w和h
            poly_init = self.use_gt_detection(output, batch) #训练的时候，这里直接使用的gt center来对点进行初始化 ，获得dla模块中推理得到的偏移量
            poly_init = poly_init.detach()
            
            #init_mask_pred = self.box_mask_head(cnn_feature, poly_init, batch['ct_01'].byte())
            if cfg.use_box:
                box_mask_pred, roi = self.box_mask_head(cnn_feature, poly_init, batch['ct_01'].byte())
                vis_mask_pred, _ = self.vis_mask_head(cnn_feature, poly_init, batch['ct_01'].byte())
                box_mask_preds.append(box_mask_pred)
                vis_mask_preds.append(vis_mask_pred)
                rois.append(roi)
            
            py_pred = poly_init * snake_config.ro  #乘了个4，对应到原图的尺寸，而他这里使用的feature map是经过4倍降采样的
            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init) #将坐标转换为相对于最左以及最上的相对坐标
            #i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],vis_mask_preds[-1][torch.arange(vis_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])  # n*64*128
                
            if cfg.use_box:
                if(cfg.use_interactive):
                    simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points(init['per_ins_cmask'],get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3])).to(init['per_ins_cmask'].device),cfg.guassian_sigma)
                    gt_boxes = self.get_box(batch['i_gt_py'][torch.arange(batch['i_gt_py'].shape[0])][batch['ct_01'].byte()])
                    scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                    simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps(scaled_boxes,(cnn_feature.shape[2],cnn_feature.shape[3]))
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],init['per_vis_cmask'].unsqueeze(1),simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                    point_guassian_heatmaps.append(simulate_points_heatmaps)
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],init['per_vis_cmask'].unsqueeze(1))
            else:
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'])
            net = torch.tanh(i_poly_fea)  ## 初始h0，就是feature aggregation得到的mid feature经过一个tanh计算
            i_poly_fea = F.leaky_relu(i_poly_fea)
            py_preds = []
            py_preds.append(py_pred)
            cls_scores= []
            for i in range(self.iter):  ## 因为初始点需要单独通过中心点来获得，因此先进行处理后，再进行迭代 ####不过他这里代码执行还是总共只执行了self.iter次迭代，因为他这里是在循环开头用gru计算偏移量的
                net, offset = self.update_block(net, i_poly_fea) # gru模块，输出net(论文中的hk)和偏移量  net送入下一轮迭代中
                #cls_score= self.classify_block(net)
                #cls_scores.append(cls_score)
                #### offset
                # offset[:,:,0]*inp_w offset[:,:,1]*=inp_h
                py_pred = py_pred + snake_config.ro * offset# * attn_score
                py_preds.append(py_pred)

                py_pred_sm = py_pred / snake_config.ro
                if cfg.use_box:
                    box_mask_pred, roi = self.box_mask_head(cnn_feature, py_pred_sm, batch['ct_01'].byte())
                    vis_mask_pred, _ = self.vis_mask_head(cnn_feature, py_pred_sm, batch['ct_01'].byte())
                    box_mask_preds.append(box_mask_pred)
                    vis_mask_preds.append(vis_mask_pred)
                    rois.append(roi)
                
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                
                #i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],vis_mask_preds[-1][torch.arange(vis_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])
                if cfg.use_box:
                    if(cfg.use_interactive):
                        simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points(init['per_ins_cmask'],get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3])).to(init['per_ins_cmask'].device), cfg.guassian_sigma)
                        point_guassian_heatmaps.append(simulate_points_heatmaps)
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],init['per_vis_cmask'].unsqueeze(1),simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                    else:
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:],init['per_vis_cmask'].unsqueeze(1))
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'])
                #attn_score = self.get_attn_score(self.boundary_coefficient, attention_feature, py_pred_sm, c_py_pred, init['py_ind'])
                i_poly_fea = F.leaky_relu(i_poly_fea)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro, 'cls_scores': cls_scores, 'mask_preds': box_mask_preds,'vis_mask_preds': vis_mask_preds,'rois': rois,'point_guassian_heatmaps':point_guassian_heatmaps})

        if not self.training:
            with torch.no_grad():
                test_box_mask_preds = []
                test_vis_mask_preds = []
                py_preds = []
                if(cfg.use_interactive):
                    init = self.prepare_training(output, more_info)
                poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3),self.score_thresh)
                device = detection.device
                # poly_init_loss = self.use_gt_detection(output, batch)
                # init = snake_gcn_utils.prepare_training(output, batch) # init中存放gt和ct对应的在batch中的图片编号
                # ret.update({'i_gt_py': init['i_gt_py']* snake_config.ro}) # 将gt加到output中保存
                ind = torch.zeros((poly_init.size(0)))
                py_preds.append(poly_init)
                py_pred = poly_init * snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init)
                ct_01 = torch.ones([1, detection.size(0)])
                if cfg.use_box:
                    box_mask_pred, roi = self.box_mask_head(cnn_feature, poly_init, ct_01.byte())
                    vis_mask_pred, _ = self.vis_mask_head(cnn_feature, poly_init, ct_01.byte())
                    test_box_mask_preds.append(box_mask_pred)
                    test_vis_mask_preds.append(vis_mask_pred)
                    if(cfg.use_interactive):
                        pred_masks=get_gt_info.points_to_mask(poly_init,(cnn_feature.shape[2],cnn_feature.shape[3]),device)
                        pred_to_gt = get_gt_info.match_masks_multiple(pred_masks.to(device), detection[:,3], more_info['per_ins_cmask'].squeeze(0).to(device),more_info['ct_cls'].squeeze(0).to(device))
                        simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points_test(more_info['per_ins_cmask'].squeeze(0).to(device),pred_masks, pred_to_gt,cfg.guassian_sigma)
                        gt_boxes = self.get_box(more_info['i_gt_py'][torch.arange(more_info['i_gt_py'].shape[0])][more_info['ct_01'].byte()])
                        scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                        simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps_test(scaled_boxes,pred_to_gt,(cnn_feature.shape[2],cnn_feature.shape[3]))
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                              ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],test_vis_mask_preds[-1][torch.arange(test_vis_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1))
                    else:
                        i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                                ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],test_vis_mask_preds[-1][torch.arange(test_vis_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:])
                else:
                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                              ind)
                #py_preds=[]
                if len(py_pred) != 0:
                    net = torch.tanh(i_poly_fea)
                    i_poly_fea = F.leaky_relu(i_poly_fea)
                    for i in range(self.iter):
                        net, offset = self.update_block(net, i_poly_fea)
                        #cls_score= self.classify_block(net)
                        py_pred = py_pred + snake_config.ro * offset
                        py_pred_sm = py_pred / snake_config.ro
                        py_preds.append(py_pred_sm)
                        if cfg.use_box:
                            box_mask_pred, roi = self.box_mask_head(cnn_feature, py_pred_sm, ct_01.byte())
                            vis_mask_pred, _ = self.vis_mask_head(cnn_feature, py_pred_sm, ct_01.byte())
                            test_box_mask_preds.append(box_mask_pred)
                            test_vis_mask_preds.append(vis_mask_pred)
                        if i != (self.iter - 1):                     
                            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                            if cfg.use_box:
                                if(cfg.use_interactive):
                                    pred_masks=get_gt_info.points_to_mask(py_pred_sm,(cnn_feature.shape[2],cnn_feature.shape[3]),device)
                                    pred_to_gt = get_gt_info.match_masks_multiple(pred_masks.to(device), detection[:,3], more_info['per_ins_cmask'].squeeze(0).to(device),more_info['ct_cls'].squeeze(0).to(device))
                                    simulate_points_heatmaps = self.iterative_sampler.generate_gaussian_heatmaps_points_test(more_info['per_ins_cmask'].squeeze(0).to(device),pred_masks, pred_to_gt, cfg.guassian_sigma)
                                    scaled_boxes = self.iterative_sampler.random_scale_boxes(gt_boxes)
                                    simulate_box_heatmaps = self.iterative_sampler.generate_smooth_rectangular_heatmaps_test(scaled_boxes,pred_to_gt,(cnn_feature.shape[2],cnn_feature.shape[3]))
                                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],test_vis_mask_preds[-1][torch.arange(test_vis_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],simulate_points_heatmaps.unsqueeze(1),simulate_box_heatmaps.unsqueeze(1)) #init['ind'])
                                else:
                                    i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:],test_vis_mask_preds[-1][torch.arange(test_vis_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:]) #init['ind'])
                            else:
                                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind) #init['ind'])
                            i_poly_fea = F.leaky_relu(i_poly_fea)
                            #attn_score = self.get_attn_score(self.boundary_coefficient, attention_feature, py_pred_sm, c_py_pred, ind)
                    final_py_preds = [py_pred_sm]
                    py_preds.append(py_pred_sm)
                    ret.update({'py': py_preds})
                    #ret.update({'amodal_preds': test_box_mask_preds,'vis_mask_preds': test_vis_mask_preds})
                else:
                    final_py_preds = [i_poly_fea]
                    ret.update({'py': final_py_preds})
        return output

    def forward(self, output, cnn_feature, batch, fine_feature, more_info = None):
        if(cfg.use_dct):
            return self.forward_with_dct(output, cnn_feature, fine_feature, batch, more_info)
            if(cfg.use_interactive):
                return self.forward_with_dct(output, cnn_feature, fine_feature, batch, more_info)
            else:
                return self.forward_with_dct(output, cnn_feature, fine_feature, batch)
        else:
            return self.forward_with_box(output, cnn_feature, batch, more_info)