import os
from lib.utils.snake import snake_voc_utils, snake_coco_utils,snake_cityscapes_coco_utils,snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from lib.config import cfg
import random


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, istrain):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        if istrain:
            self.coco = COCO(ann_file[0])
        else:
            self.coco = COCO(ann_file)
        self.anns = sorted(self.coco.getImgIds())
        if istrain:
            self.coco_aug=COCO(ann_file[1])
            self.anns=sorted(self.anns+self.coco_aug.getImgIds())
        self.istrain=istrain
        if istrain:
            self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=0)) or len(self.coco_aug.getAnnIds(imgIds=ann, iscrowd=0))])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        if(len(self.coco.getAnnIds(imgIds=img_id, iscrowd=0))):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)
            anno=self.coco.loadAnns(ann_ids)
            path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        else:
            ann_ids = self.coco_aug.getAnnIds(imgIds=img_id, iscrowd=0)
            anno=self.coco_aug.loadAnns(ann_ids)
            path = os.path.join(self.data_root, self.coco_aug.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def read_original_data(self, anno, path):
        # print(path)
        assert os.path.exists(path)
        img = cv2.imread(path)
        #if(type(anno[0]['segmentation'])==dict):
        for obj in anno:
            if(type(obj['segmentation'])!=dict):
                continue
            seg_contour=[]
            t=mask_util.decode(obj['segmentation'])      
            poly=data_utils.polygonFromMask(t)
            if(len(poly)==0):
                print("errno")
            for p in poly:
                seg_contour.append(p)
            obj['segmentation']=seg_contour
        try:
            if not self.istrain:  ## 这里使用的数据集为原装coco_amodal_val2014.json 但这里的segmentation
                #instance_polys = [[[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']]for obj in pic['regions']] for pic in anno]
                instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
            else:
                instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        except:
            print("debug point")
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

            polys = snake_coco_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_coco_utils.filter_tiny_polys(instance)
            polys = snake_coco_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_coco_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    # def prepare_detection(self, box, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind):
    def prepare_detection(self, box, poly, ct_hm, cls_id, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        # ct_float = ct.copy()
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        # wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        # reg.append((ct_float  - ct).tolist())

        # x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        # x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        # decode_box = [x_min, y_min, x_max, y_max]
        #
        # return decode_box

    # def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
    #     x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
    #     x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
    #
    #     img_init_poly = snake_coco_utils.get_init(box)
    #     img_init_poly = snake_coco_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
    #     can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
    #     img_gt_poly = extreme_point
    #     can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)
    #
    #     i_it_4pys.append(img_init_poly)
    #     c_it_4pys.append(can_init_poly)
    #     i_gt_4pys.append(img_gt_poly)
    #     c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, img_gt_polys):
        img_gt_poly = snake_voc_utils.uniformsample(poly, len(poly) * 128)
        idx = self.four_idx(img_gt_poly)
        img_gt_poly = self.get_img_gt(img_gt_poly, idx)
        img_gt_polys.append(img_gt_poly)

    def four_idx(self, img_gt_poly):
        x_min, y_min = np.min(img_gt_poly, axis=0)
        x_max, y_max = np.max(img_gt_poly, axis=0)
        center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
        can_gt_polys = img_gt_poly.copy()
        can_gt_polys[:, 0] -= center[0]
        can_gt_polys[:, 1] -= center[1]
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
        idx_bottom = np.argmax(can_gt_polys[:, 1])
        idx_top = np.argmin(can_gt_polys[:, 1])
        idx_right = np.argmax(can_gt_polys[:, 0])
        idx_left = np.argmin(can_gt_polys[:, 0])
        return [idx_bottom, idx_right, idx_top, idx_left]

    def get_img_gt(self, img_gt_poly, idx, t=128):
        align = len(idx)
        pointsNum = img_gt_poly.shape[0]
        r = []
        k = np.arange(0, t / align, dtype=float) / (t / align)
        for i in range(align):
            begin = idx[i]
            end = idx[(i + 1) % align]
            if begin > end:
                end += pointsNum
            r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        r = np.concatenate(r, axis=0)
        return img_gt_poly[r, :]

    def img_poly_to_can_poly(self, img_poly):
        x_min, y_min = np.min(img_poly, axis=0)
        can_poly = img_poly - np.array([x_min, y_min])
        return can_poly

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_coco_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # extreme_points = self.get_extreme_points(instance_polys)

        # detection
        output_h, output_w = inp_out_hw[2:]
        # print(cfg.heads.ct_hm)
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        # wh = []
        # reg = []
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
        
        cmask = snake_voc_utils.polygon_to_cmask(instance_polys, output_h, output_w)[np.newaxis,:,:]
        # c_gt_pys = []

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

                # self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind)
                self.prepare_detection(bbox, poly, ct_hm, cls_id, ct_cls, ct_ind)
                # self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                # self.prepare_evolution(bbox, poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys, inp_out_hw)
                self.prepare_evolution(poly, i_gt_pys)

        ret = {'inp': inp, 'cmask': cmask}
        # detection = {'ct_hm': ct_hm, 'wh': wh, 'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        # init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        # evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        # i_gt_pys[:,0]=i_gt_pys[:,0]/output_w
        # i_gt_pys[:,1]=i_gt_pys[:,1]/output_h
        evolution = {'i_gt_py': i_gt_pys}
        ret.update(detection)
        # ret.update(init)
        ret.update(evolution)
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num, 'aug_img': orig_img, 'path': path, 'inp_out_hw':inp_out_hw}

        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

