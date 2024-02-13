import torch.nn as nn
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net,isTrain=True):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.isTrain=isTrain
        self.m_crit = net_utils.FocalLoss() ## mask loss
        self.ct_crit = net_utils.FocalLoss() ## center loss
        self.cls_crit = net_utils.ClsCrossEntropyLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss  ## poly loss
    
    def shape_loss(self, pred, targ_shape):
        pre_dis = torch.cat((pred[:,1:], pred[:,0].unsqueeze(1)), dim=1)
        pred_shape = pre_dis-pred
        # targ_shape = targ(:,:,1:)-targ(:,:,:-1)
        loss = self.py_crit(pred_shape, targ_shape)
        return loss

    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        ct_01=batch['ct_01'].byte()
        scalar_stats = {}
        loss = 0
        ## 下面使用的net_utils.sigmoid，经过sigmoid后，将最小值设定为了1e-4,最大值设定为了1-1e-4
        mask_loss = self.m_crit(net_utils.sigmoid(output['mask']), batch['cmask'])  ## 预测的boundary mask loss ,用的focal loss
        scalar_stats.update({'mask_loss': mask_loss})
        loss += mask_loss
        
        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])  ##直接计算ct_hm的loss，这样对所有点都计算了个loss，这里因为加了一个高斯模糊核，在左上和右下一定范围内都一定程度减少了损失
        #ct_loss = self.ct_crit(output['ct_hm'], batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.py_crit(output['poly_init'], output['i_gt_py'])  ## 初始点的点坐标loss使用smooth-l1 loss，与目标检测一样
        scalar_stats.update({'wh_loss': 0.1 * wh_loss})  ## 降低了这部分loss的梯度
        loss += 0.1 * wh_loss

        n_predictions = len(output['py_pred'])  ## 将每次迭代的预测点都计算一次loss
        py_loss = 0.0
        shape_loss = 0.0
        #cls_loss = 0.0
        py_dis = torch.cat((output['i_gt_py'][:,1:], output['i_gt_py'][:,0].unsqueeze(1)), dim=1)  ## 将i_gt_py整体循环左移方便后面求距离
        tar_shape = py_dis - output['i_gt_py']  ##得出gt点的相对的偏移量
        for i in range(n_predictions):
            i_weight = 0.8**(n_predictions - i - 1)  ### 这个权重越靠近后预测的权重越大,越靠后的应该权重越大，预测错误的话惩罚应该给的更大
            py_loss += i_weight * self.py_crit(output['py_pred'][i], output['i_gt_py'])
            shape_loss += i_weight * self.shape_loss(output['py_pred'][i], tar_shape)
            #cls_loss += self.cls_crit(output['cls_scores'][i], batch['ct_cls'][ct_01])

        py_loss = py_loss / n_predictions  ## loss均分，与cascade中一样，防止过度训练，而且因为后续迭代的loss也会影响到之前的参数训练
        shape_loss = shape_loss / n_predictions
        #cls_loss = cls_loss / n_predictions
        scalar_stats.update({'py_loss': py_loss})
        scalar_stats.update({'shape_loss': shape_loss})
        #scalar_stats.update({'cls_loss': cls_loss})
        loss += py_loss
        loss += shape_loss
        #loss += cls_loss
        # condinst loss
        #per_ins_cmask = torch.cat([batch['per_ins_cmask'][i][ct_01[i]] for i in range(ct_01.size(0))], dim=0)
        # cond_mask_loss = self.m_crit(net_utils.sigmoid(output['cond_predict'][0]), per_ins_cmask)
        cond_mask_loss = net_utils.dice_coefficient(net_utils.sigmoid(output['cond_predict']), output['per_ins_cmask'])
        cond_mask_loss = cond_mask_loss.mean()
        scalar_stats.update({'cond_mask_loss': cond_mask_loss})
        loss += cond_mask_loss
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats