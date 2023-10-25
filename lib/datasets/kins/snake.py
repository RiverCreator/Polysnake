import torch.utils.data as data
from lib.utils.snake import snake_kins_utils, snake_voc_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.config import cfg
import pycocotools.mask as mask_util
import random


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, istrain):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.istrain=istrain
        self.coco = COCO(ann_file)
        self.anns = np.array(self.coco.getImgIds())
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())} ## 由于标注中并不是连续的id，这里重新映射为连续的id

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        if(type(anno[0]['segmentation'])==dict):
            for obj in anno:
                seg_contour=[]
                t=mask_util.decode(obj['segmentation'])       
                poly=data_utils.polygonFromMask(t)
                if(len(poly)==0):
                    print("errno")
                for p in poly:
                    seg_contour.append(p)
                obj['segmentation']=seg_contour
        # if not self.istrain:
        #     print("debug point")
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno   ## [[[]]] instance_polys[0][0][0]为array([690,182])
                          if not isinstance(obj['segmentation'], dict)]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = instance

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_kins_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = snake_kins_utils.filter_tiny_polys(instance)
            polys = snake_kins_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_kins_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32) #box center
        ct = np.round(ct).astype(np.int32) ## 取整

        h, w = y_max - y_min, x_max - x_min  ##获得object的外界矩阵长宽
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w))) ## 求得物体框的高斯半径，也即划分为正样本的一个边界范围
        radius = max(0, int(radius)) 
        data_utils.draw_umich_gaussian(ct_hm, ct, radius) ## 通过高斯半径求出一个对应的高斯散射核，让gt heatmap向周围方向逐渐缓和，中心点的标签为1，周围逐渐缓和，这里主要用到center上，gt center不只是一个点，而周围一定范围内都可以是，但是标签并不是1.

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0]) ## 从左往右从上往下，一个像素点一个编号，求出center的编号

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        # return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_kins_utils.get_init(box)
        img_init_poly = snake_kins_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    # def prepare_evolution(self, bbox, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        # x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        # x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        
        if random.random() > 0.33:
            aug_para = (np.random.rand(4)) * 3 - 1.5
            bbox = list(np.array(bbox) + aug_para)
            bbox = np.clip(bbox, 0, 224)
            bbox[3] = np.clip(bbox[3], 0, 96)
            
        # extreme_point = snake_kins_utils.get_init(bbox)

        # octagon = snake_kins_utils.get_octagon(extreme_point)
        # img_init_poly = snake_kins_utils.uniformsample(octagon, snake_config.poly_num)
        # can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        # img_gt_poly = snake_kins_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        # tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        # img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        # can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        # img_init_polys.append(img_init_poly)
        # can_init_polys.append(can_init_poly)
        # img_gt_polys.append(img_gt_poly)
        # can_gt_polys.append(can_gt_poly)

    # def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        # cp_id.append(is_id)
        # cp_cls.append(cls_id)
        
    def prepare_evolution(self, poly, img_gt_polys):
        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * 128) #多采样为128倍的点数
        idx = self.four_idx(img_gt_poly) #
        img_gt_poly = self.get_img_gt(img_gt_poly, idx)
        img_gt_polys.append(img_gt_poly)

    def four_idx(self, img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.] #计算出box的中心
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0] # 这里计算相对于center的偏移量
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6    ##计算出点到中心的距离
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2) ## 坐标除以距离，将坐标归一化到-1到1之间
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]  ##这里的算法应该是E2EC中的MDA，只有径直方向的坐标最大，接近1，找到四个关键位置

    def get_img_gt(self, img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)  ## 在四个关键点之间均匀采样，确定比例
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align] #起始点和结束点
            if begin > end: #循环了 结束点在循环起点后，为了方便计算关键点之间的点个数
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum) #将对应比例位置的点编号计算放到r中
        r = np.concatenate(r, axis=0) #四个关键点之间四个区域
        return img_gt_poly[r, :] #将这些采样点取出

    def img_poly_to_can_poly(self, img_poly):
        x_min, y_min = np.min(img_poly, axis=0)
        can_poly = img_poly - np.array([x_min, y_min])
        return can_poly
        
    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)  ### anno[0]['segmentation']=[[]]
        ## img为对应原图数据，instance_polys为标注中的segmentation，数量不统一，cls_ids为对应分割实例的类别
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_kins_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)  #经过数据增强后还需要对poly points及逆行仿射变换
        instance_polys = self.get_valid_polys(instance_polys)
        # extreme_points = self.get_extreme_points(instance_polys)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        # # init
        # i_it_4pys = []
        # c_it_4pys = []
        # i_gt_4pys = []
        # c_gt_4pys = []

        # evolution
        # i_it_pys = []
        # c_it_pys = []
        i_gt_pys = []
        # c_gt_pys = []

        cmask = snake_voc_utils.polygon_to_cmask(instance_polys, output_h, output_w)[np.newaxis,:,:]  ## 将polygon转换为mask
        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            # instance_points = extreme_points[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                # extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                # self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                # self.prepare_evolution(bbox, poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
                self.prepare_evolution(poly, i_gt_pys) ## poly为数据集中的poly点，这里将其上采样为128个，存放到i_gt_pys里面
                
        # ret = {'inp': inp, 'cmask': cmask}
        # detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        # evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        # ret.update(detection)
        # ret.update(evolution)

        # ct_num = len(ct_ind)
        # meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        # ret.update({'meta': meta})
        
        ret = {'inp': inp, 'cmask': cmask}
        detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        evolution = {'i_gt_py': i_gt_pys}
        ret.update(detection)
        ret.update(evolution)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num, 'path': path}
        ret.update({'meta': meta})
        return ret

    def __len__(self):
        return len(self.anns)

