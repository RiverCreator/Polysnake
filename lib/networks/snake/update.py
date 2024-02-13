import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv1d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1))) 

        h = (1-z) * h + z * q
        
        return h


class BasicUpdateBlock(nn.Module):
    def __init__(self):
        super(BasicUpdateBlock, self).__init__()
        self.gru = ConvGRU(hidden_dim=64, input_dim=64)

        self.prediction = nn.Sequential(   ## FFN
            nn.Conv1d(64, 128, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 2, 1)
            )

    def forward(self, net, i_poly_fea):
        net = self.gru(net, i_poly_fea)
        offset = self.prediction(net).permute(0, 2, 1)

        return net, offset
class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(x):
        pass

class ClassifyBlock(nn.Module):
    def __init__(self, mid_channel = 1024, num_classes = 60):
        super(ClassifyBlock,self).__init__()
        # input center_num 64 128
        self.predication=nn.Sequential(
            nn.Linear(64*128,mid_channel),
            nn.BatchNorm1d(mid_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mid_channel,num_classes)
        )
    
    def forward(self, x):
        x=x.view(x.shape[0], x.shape[1]*x.shape[2])
        scores = self.predication(x)
        return scores

class OcclusionAtte(nn.Module):
    def __init__(self):
        super(OcclusionAtte,self).__init__()
        conv1 = Conv2d(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=nn.GroupNorm(4, 32),
                activation=F.relu,
        )
        conv2 = Conv2d(
            32,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(4, 32),
            activation=F.relu,
        )
        conv3 = Conv2d(
            32,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation=F.relu,
        )
        weight_init.c2_msra_fill(conv1)
        weight_init.c2_msra_fill(conv2)
        nn.init.normal_(conv3.weight, 0, 0.01)
        nn.init.constant_(conv3.bias, 0)
        self.attender = nn.Sequential(conv1, conv2, conv3)

    def forward(self,x):
        return self.attender(x)
    