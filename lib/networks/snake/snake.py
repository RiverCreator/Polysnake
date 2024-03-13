from turtle import forward
import torch.nn as nn
import torch


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.LeakyReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, need_fea, conv_type='dgrid'):
        super(Snake, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim, conv_type) 

        self.res_layer_num = 7 # 这里加上head，共有八层循环卷积，这里使用的dgrid为dilated circular conv，一定程度增加了感受野，卷积核大小为9
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i]) #这里让输入输出维度保持不变，这样来方便进行残差链接
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1) ## circular convolution  
        if need_fea: ## 不直接预测，而是得到中间的feature（64维）
            self.prediction = nn.Sequential(
                nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(256, 64, 1)
            )
        else:
            self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
            )

    def forward(self, x):
        states = []
        x = self.head(x)  ## x(n 64+2 128) -> x(n 128 128)
        states.append(x)
        for i in range(self.res_layer_num): ## 7层conv1d 保持形状不变 
            x = self.__getattr__('res'+str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1) ## 将每一次卷积的结果进行concat 得到(n 128*8 128)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]  ##fusion结果转换为256维的(n 256 128)  这里求max返回的是value indices，[0]即返回的是value（n 256 1）相当于是把所有点对应channel进行求max，取出每个channel中最为突出的点
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2)) ## 求完max后，重新扩张为（n 256 128）
        state = torch.cat([global_state, state], dim=1)  ### 将一个全局信息和所有信息进行concat 得到(n,1024+256,128)
        x = self.prediction(state)  # 这里得到的并不是预测，而是mid feature，得到（n,64,128），论文中将这个feature送到gru中进行迭代
        return x

class BEB(nn.Module):
    #boundary envolve block
    def __init__(self, state_dim, feature_dim, conv_type= 'dgrid'):
        super(BEB, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim, conv_type)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim , 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.head(x)
        x = self.prediction(x)
        return x

class Poly_Fusion(nn.Module):
    def __init__(self, feature_dim, conv_type= 'dgrid') -> None:
        super(Poly_Fusion, self).__init__()
        #feature dim为输入的dim，state dim为输出的dim
        self.headx = BasicBlock(feature_dim, feature_dim, conv_type)
        self.heady = BasicBlock(feature_dim, feature_dim, conv_type)
    
    def forward(self, x, y):
        x = self.headx(x)
        y = self.heady(y)
        return x + y