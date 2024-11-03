import numpy as np
import torch
from .dct import dct_2d,idct_2d


class DctMaskEncoding(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, vec_dim, mask_size=128):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        assert vec_dim <= mask_size*mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, masks, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim] #取前vec_dim个
        else:
            dct_vector_coords = self.dct_vector_coords[:dim] #这个坐标是zigzag形的，取低频前dim个feature
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=float)  # [N, H, W]
        dct_all = dct_2d(masks, norm='ortho') #本来最大值是64（patch_size*patch_size）ortho归一化会让每个维度的 DCT 结果都会乘以 1 / sqrt(8) 因此最大值为64/8=8就是patch_size
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector 每个patch选出低频的信息过滤额外的信息
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device

        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = idct_2d(dct_trans, norm='ortho')  # [N, mask_size, mask_size]
        return mask_rc

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs
