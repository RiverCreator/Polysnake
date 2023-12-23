import torch.nn as nn
import torch

from .dla import DLASeg
from .ICD import RAFT


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1, 
                          last_level=5,
                          head_conv=head_conv)
        self.raft = RAFT() ##Recurrent all-pairs Field transforms 也就是论文中进行迭代回归的模块

    def forward(self, x, batch=None):
        # batch 主要是保存gt信息的，共有inp，meta，ct_hm,cmask,ct_cls,ct_ind,ct_01,ct_img_idx,i_gt_py
        ## inp指的是经过数据增强、裁剪、归一化和norm后的input(这里调用forward的时候，x即batch['inp'])，ct_hm指的是中心点的gt heatmap，具体解释看datasets/kins/snake.py中的Dataset类prepare detection函数
        # i_gt_py指的采样的128个gt点，cmask为polygon转换来的mask， ct_ind为center所在的编号（像素点位置，从左到右从上到下） meta中的数据为标注中的数据
        # ct_01和ct_img_idx为用来检测中的，具体的ct_01表示中每一行，从0到ct_num下标都为1，ct_img_idx每一行表示 从0到ct_num下标为对应的batch中的idx。
        output, cnn_feature = self.dla(x) # dla输出的output为ct_heatmap,point wh offset,boundry mask,这里输出的cnn_feature为x经过4倍降采样的(而且是经过多个尺度特征增强的，)
        output = self.raft(output, cnn_feature, batch) #这里的batch
        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
