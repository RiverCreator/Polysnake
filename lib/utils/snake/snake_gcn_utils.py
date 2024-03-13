import torch
import numpy as np
from lib.utils.snake import snake_decode, snake_config
from lib.csrc.extreme_utils import _ext as extreme_utils
from lib.utils import data_utils


def collect_training(poly, ct_01): #poly即i_gt_pys,
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0) ## 这里只提取到前ct_num（也就是对应的预测的所有的ct）个ct的poly
    return poly

def collect_ins_training(per_ins_mask, ct_01):
    batch_size = ct_01.size(0)
    per_ins_mask = torch.cat([per_ins_mask[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return per_ins_mask

def prepare_training_init(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}
    init.update({'i_it_4py': collect_training(batch['i_it_4py'], ct_01)})
    init.update({'c_it_4py': collect_training(batch['c_it_4py'], ct_01)})
    init.update({'i_gt_4py': collect_training(batch['i_gt_4py'], ct_01)})
    init.update({'c_gt_4py': collect_training(batch['c_gt_4py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    return init


# def prepare_testing_init(box, score):
    # i_it_4pys = snake_decode.get_init(box)
    # i_it_4pys = uniform_upsample(i_it_4pys, snake_config.init_poly_num)
    # c_it_4pys = img_poly_to_can_poly(i_it_4pys)

    # ind = score > snake_config.ct_score
    # i_it_4pys = i_it_4pys[ind]
    # c_it_4pys = c_it_4pys[ind]
    # ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    # init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'ind': ind}

    # return init


def prepare_testing_init(score):
    #print('ind', score.shape)  # ind torch.Size([100])
    ind = score > snake_config.ct_score
    # print('snake_config.ct_score', snake_config.ct_score)  # snake_config.ct_score 0.05
    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    init = {'ind': ind}
    # print(ind)
    
    return init
    

def get_box_match_ind(pred_box, score, gt_poly):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > snake_config.box_iou) * (score > snake_config.confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind]

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1]
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind


def prepare_training_box(ret, batch, init):
    box = ret['detection'][..., :4]
    score = ret['detection'][..., 4]
    batch_size = box.size(0)
    i_gt_4py = batch['i_gt_4py']
    ct_01 = batch['ct_01'].byte()
    ind = [get_box_match_ind(box[i], score[i], i_gt_4py[i][ct_01[i]]) for i in range(batch_size)]
    box_ind = [ind_[0] for ind_ in ind]
    gt_ind = [ind_[1] for ind_ in ind]

    i_it_4py = torch.cat([snake_decode.get_init(box[i][box_ind[i]][None]) for i in range(batch_size)], dim=1)
    if i_it_4py.size(1) == 0:
        return

    i_it_4py = uniform_upsample(i_it_4py, snake_config.init_poly_num)[0]
    c_it_4py = img_poly_to_can_poly(i_it_4py)
    i_gt_4py = torch.cat([batch['i_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    c_gt_4py = torch.cat([batch['c_gt_4py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_4py = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'i_gt_4py': i_gt_4py, 'c_gt_4py': c_gt_4py}

    i_it_py = snake_decode.get_octagon(i_gt_4py[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    i_gt_py = torch.cat([batch['i_gt_py'][i][gt_ind[i]] for i in range(batch_size)], dim=0)
    init_py = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    ind = torch.cat([torch.full([len(gt_ind[i])], i) for i in range(batch_size)], dim=0)

    if snake_config.train_pred_box_only:
        for k, v in init_4py.items():
            init[k] = v
        for k, v in init_py.items():
            init[k] = v
        init['4py_ind'] = ind
        init['py_ind'] = ind
    else:
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_4py.items()})
        init.update({'4py_ind': torch.cat([init['4py_ind'], ind], dim=0)})
        init.update({k: torch.cat([init[k], v], dim=0) for k, v in init_py.items()})
        init.update({'py_ind': torch.cat([init['py_ind'], ind], dim=0)})


def prepare_training(ret, batch): ## gt存放到init中，并且标记每个ct对应batch中的第几个图片
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})  ## 每个ct对应的gt点
    init.update({'per_ins_cmask': collect_ins_training(batch['per_ins_cmask'], ct_01)})
    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})  ## py_ind用来表示ct对应的是batch中哪一张图片

    init['py_ind'] = init['py_ind'].to(ct_01.device)  

    return init


def prepare_training_evolve(ex, init):
    if not snake_config.train_pred_ex:
        evolve = {'i_it_py': init['i_it_py'], 'c_it_py': init['c_it_py'], 'i_gt_py': init['i_gt_py']}
        return evolve

    i_gt_py = init['i_gt_py']

    if snake_config.train_nearest_gt:
        shift = -(ex[:, :1] - i_gt_py).pow(2).sum(2).argmin(1)
        i_gt_py = extreme_utils.roll_array(i_gt_py, shift)

    i_it_py = snake_decode.get_octagon(ex[None])
    i_it_py = uniform_upsample(i_it_py, snake_config.poly_num)[0]
    c_it_py = img_poly_to_can_poly(i_it_py)
    evolve = {'i_it_py': i_it_py, 'c_it_py': c_it_py, 'i_gt_py': i_gt_py}

    return evolve


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        #print(ex.shape)  # 1*4*2
        i_it_pys = snake_decode.get_octagon(ex[None])
        #print(i_it_pys.shape) # 1 * 1 * 12 * 2
        i_it_pys = uniform_upsample(i_it_pys, snake_config.poly_num)[0]
        #print(i_it_pys.shape)  # 1 * 128 * 2
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1 ## x坐标
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1 ## y坐标 为什么要w/2和h/2? 这里是为了配合后面grid_sample函数，其第二个参数即采样坐标，但坐标需要norm到（-1,1）之间 这里除以w/2和h/2，将其norm到(0,2) 减一norm到(-1,1)
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)  ## ct_num cnn_feature_channel poly_points_num 
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0) ## 选出第i张图片的center的init points 转为（1，ct_num,128,2）
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2) ## cnn_feature[i:i+1]对应第i张图片的cnn_feature，如果坐标超出范围则直接取0，因为该函数padding默认参数为zeros，如果为border，则用边缘元素填充
        gcn_feature[ind == i] = feature

    return gcn_feature

def get_mask_probility(mask_preds, norm_poly):
    # mask_preds [n,1,28,28] poly [n,128,2]
    polys = norm_poly.clone()
    h ,w = mask_preds.size(2),mask_preds.size(3)
    polys[..., 0] = polys[..., 0] / (w / 2.) -1
    polys[..., 1] = polys[..., 1] / (h / 2.) -1
    n = mask_preds.size(0)
    probs = torch.zeros([n, 1, polys.size(1)]).to(polys.device) #(n,1,128)
    for i in range(n):
        poly = polys[i].unsqueeze(0).unsqueeze(0)
        prob = torch.nn.functional.grid_sample(mask_preds[i:i+1], poly)[0].permute(1, 0, 2)
        probs[i] = prob
    return probs

def get_adj_mat(n_adj, n_nodes, device):
    a = np.zeros([n_nodes, n_nodes])

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1

    a = torch.Tensor(a.astype(np.float32))
    return a.to(device)


def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def get_pconv_ind(n_adj, n_nodes, device):
    n_outer_nodes = snake_config.poly_num
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1)])
    outer_ind = (torch.arange(n_outer_nodes)[:, None] + ind[None]) % n_outer_nodes
    inner_ind = outer_ind + n_outer_nodes
    ind = torch.cat([outer_ind, inner_ind], dim=1)
    return ind


def img_poly_to_can_poly(img_poly):
    ## 将所有坐标减去最小值的坐标，这样让坐标的值域为0以上，避免了绝对位置的改变
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    # x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    # y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = torch.max(h, w)
    # can_poly = can_poly / long_side[..., None, None]
    return can_poly


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def zoom_poly(poly, scale):
    mean = (poly.min(dim=1, keepdim=True)[0] + poly.max(dim=1, keepdim=True)[0]) * 0.5
    poly = poly - mean
    poly = poly * scale + mean
    return poly

def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes): A Boxes of length Bimg. boxes.tensor[i] and masks[i] correspond
            to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)

    boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks