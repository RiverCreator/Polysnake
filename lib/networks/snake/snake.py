from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
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

        self.res_layer_num = 5 # 这里加上head，共有八层循环卷积，这里使用的dgrid为dilated circular conv，一定程度增加了感受野，卷积核大小为9
        #dilation = [1, 1, 1, 2, 2, 4, 4]
        dilation = [1, 2, 2, 4, 4]
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
    
def create_expanded_adj_matrix(num_points, k):
    """
    创建每个点连接周围 k 个点的邻接矩阵（环形）
    :param num_points: 点的数量
    :param k: 每个点连接周围的 k 个点
    :return: 邻接矩阵 (N, N)
    """
    adj = torch.zeros((num_points, num_points), dtype=torch.float32)
    
    for i in range(num_points):
        for j in range(-k, k + 1):  # 包含自身和周围 k 个点
            neighbor = (i + j) % num_points  # 环形索引
            adj[i, neighbor] = 1
    
    return adj

    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                n_heads: int, concat: bool = False, dropout: float = 0.4,
                leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate

        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.permute(0, 1, 3, 2)
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        batch = h.shape[0]
        n_nodes = h.shape[2]
        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        h_transformed = torch.matmul(h.transpose(1,2), self.W).transpose(1,2)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.contiguous().view(batch, n_nodes, self.n_heads, self.n_hidden).permute(0, 2, 1, 3)
            
        # getting the attention scores
        # output shape (n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        
        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (n_nodes, out_features)
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch, n_nodes, self.out_features)
        else:
            #h_prime = h_prime.mean(dim=0)
            return h_prime

        return h_prime
    
class GAT(nn.Module):
    def __init__(self,
        in_features,
        n_hidden,
        out_features,
        n_heads,
        concat=False,
        dropout=0.0,
        leaky_relu_slope=0.2):

        super(GAT, self).__init__()
        self.adj_mat = create_expanded_adj_matrix(128, 7)
        # Define the Graph Attention layers
        self.conv_head = Snake(state_dim=128, feature_dim=in_features, conv_type='dgrid', need_fea=True)
        # self.gat1 = GraphAttentionLayer(
        #     in_features=64, out_features=n_hidden, n_heads=n_heads,
        #     concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        #     )
        
        # self.gat2 = GraphAttentionLayer(
        #     in_features=n_hidden, out_features=n_hidden, n_heads=n_heads,
        #     concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        #     )

        # self.gat3 =  GraphAttentionLayer(
        #     in_features=n_hidden, out_features=out_features, n_heads=1,
        #     concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        #     )  
    def forward(self, input_tensor: torch.Tensor):
        self.adj_mat = self.adj_mat.to(input_tensor.device)
        # Apply the first Graph Attention layer
        x = self.conv_head(input_tensor)
        # x = self.gat1(x, self.adj_mat).permute(0 ,2 ,1)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer

        # Apply the second Graph Attention layer
        # x = self.gat2(x, self.adj_mat).permute(0, 2, 1)
        # x = F.elu(x)
        
        # x = self.gat3(x, self.adj_mat).squeeze(1).permute(0, 2, 1)
        #x = x.squeeze(1).permute(0, 2, 1)
        return x # Apply softmax activation function
    
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