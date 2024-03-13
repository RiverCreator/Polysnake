from matplotlib.pyplot import box
from .snake import Snake,BEB
from .update import BasicUpdateBlock, ClassifyBlock,OcclusionAtte
from lib.utils import data_utils
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .ABranch import AmodalBranch

class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.iter = cfg.iter_num  # iteration number
        self.score_thresh=cfg.score_thresh
        #这里的state_dim表示128个点的特征向量
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64 + 2 + 1, conv_type='dgrid', need_fea=True) #即文章中用来进行特征聚合，然后输出g_{k-1}的模块
        self.update_block = BasicUpdateBlock() ## 即文章中使用gru的模块
        self.box_mask_head = AmodalBranch(cfg.num_classes)
        #self.classify_block= ClassifyBlock(1024, cfg.num_classes)
        #self.mcr=Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', need_fea=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch) # init中存放gt和ct对应的在batch中的图片编号
        output.update({'i_gt_py': init['i_gt_py'], 'per_ins_cmask': init['per_ins_cmask']}) # 将gt加到output中保存
        return init

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, box_pred):  # i_it_poly为init point，c_it_poly为相对init point，ind为标注ct为batch中哪个图片的
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)  ## cnn_featuer为b c h w  i_it_poly为(n,128,2),n为center个数
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  ### 将坐标对应的feature进行采样，每个点对应的feature即为长度为c的向量，故init_feature大小为n c 128，n为center个数，c为cnn_feature的channel
        x_min = torch.min(i_it_poly[..., 0], dim=-1)[0]
        y_min = torch.min(i_it_poly[..., 1], dim=-1)[0]
        x_max = torch.max(i_it_poly[..., 0], dim=-1)[0]
        y_max = torch.max(i_it_poly[..., 1], dim=-1)[0]
        ins_h = y_max - y_min
        ins_w = x_max - x_min
        tx = (c_it_poly[..., 0]/ins_w[:, None]* cfg.roi_w)
        ty = (c_it_poly[..., 1]/ins_h[:, None]* cfg.roi_h)
        relative_box_poly = torch.stack((tx,ty),dim=2)
        probs = snake_gcn_utils.get_mask_probility(box_pred, relative_box_poly)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1), probs.sigmoid()], dim=1)  ## 论文中提到的将相对坐标信息与之concat，提供一个相对坐标信息 c_it_poly为（n，128，2），为了能够匹配上将其转换为n 2 128 这样最终feature 大小为n c+2 128
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
    
    
    def forward(self, output, cnn_feature, batch):
        #boundary_score=output['mask'].sigmoid()
        #attention_feature = self.occlusionatte(1-boundary_score)
        box_mask_preds = []
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
            box_mask_pred, roi = self.box_mask_head(cnn_feature, poly_init, batch['ct_01'].byte())
            box_mask_preds.append(box_mask_pred)
            rois.append(roi)
            
            py_pred = poly_init * snake_config.ro  #乘了个4，对应到原图的尺寸，而他这里使用的feature map是经过4倍降采样的
            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init) #将坐标转换为相对于最左以及最上的相对坐标
            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])  # n*64*128
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
                box_mask_pred, roi = self.box_mask_head(cnn_feature, py_pred_sm, batch['ct_01'].byte())
                box_mask_preds.append(box_mask_pred)
                rois.append(roi)
                
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, init['py_ind'], box_mask_preds[-1][torch.arange(box_mask_preds[-1].shape[0]),batch['ct_cls'][batch['ct_01'].byte()]][:,None,:,:])
                #attn_score = self.get_attn_score(self.boundary_coefficient, attention_feature, py_pred_sm, c_py_pred, init['py_ind'])
                i_poly_fea = F.leaky_relu(i_poly_fea)
            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro, 'cls_scores': cls_scores, 'mask_preds': box_mask_preds, 'rois': rois})

        if not self.training:
            with torch.no_grad():
                test_box_mask_preds = []
                poly_init, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3),self.score_thresh)
                # poly_init_loss = self.use_gt_detection(output, batch)
                # init = snake_gcn_utils.prepare_training(output, batch) # init中存放gt和ct对应的在batch中的图片编号
                # ret.update({'i_gt_py': init['i_gt_py']* snake_config.ro}) # 将gt加到output中保存
                ind = torch.zeros((poly_init.size(0)))
                py_pred = poly_init * snake_config.ro
                c_py_pred = snake_gcn_utils.img_poly_to_can_poly(poly_init)
                ct_01 = torch.ones([1, detection.size(0)])
                box_mask_pred, roi = self.box_mask_head(cnn_feature, poly_init, ct_01.byte())
                test_box_mask_preds.append(box_mask_pred)
                i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, poly_init, c_py_pred,
                                              ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:])
                
                #py_preds=[]
                if len(py_pred) != 0:
                    net = torch.tanh(i_poly_fea)
                    i_poly_fea = F.leaky_relu(i_poly_fea)
                    for i in range(self.iter):
                        net, offset = self.update_block(net, i_poly_fea)
                        #cls_score= self.classify_block(net)
                        py_pred = py_pred + snake_config.ro * offset
                        #py_preds.append(py_pred)
                        py_pred_sm = py_pred / snake_config.ro
                        box_mask_pred, roi = self.box_mask_head(cnn_feature, py_pred_sm, ct_01.byte())
                        test_box_mask_preds.append(box_mask_pred)
                        if i != (self.iter - 1):                     
                            c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred_sm)
                            i_poly_fea = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred_sm, c_py_pred, ind, test_box_mask_preds[-1][torch.arange(test_box_mask_preds[-1].shape[0]),detection[:,3].long()][:,None,:,:]) #init['ind'])
                            i_poly_fea = F.leaky_relu(i_poly_fea)
                            #attn_score = self.get_attn_score(self.boundary_coefficient, attention_feature, py_pred_sm, c_py_pred, ind)
                    final_py_preds = [py_pred_sm]
                else:
                    final_py_preds = [i_poly_fea]
                ret.update({'py': final_py_preds})
        return output

