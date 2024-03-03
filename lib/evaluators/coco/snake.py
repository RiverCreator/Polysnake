import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
import torch
from PIL import Image,ImageDraw
import shutil

class Evaluator:
    def __init__(self, result_dir, logg):
        self.logger = logg
        self.results = []
        self.img_ids = []
        self.aps = []
        self.i=0
        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.threshold = 0.3
        
    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        # cond_ins_mask=output['cond_predict_val'].sigmoid()
        # cond_ins_mask_t = np.asarray(cond_ins_mask.cpu())
        #py_init=output['poly_init_infer'][output['idx']][output['nms_keep']].cpu().numpy()
        #py=py_init
        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy() #d2sa:(1952,1504)
        
        
        h, w = batch['inp'].size(2), batch['inp'].size(3)
        # py[:,:,0]=py[:,:,0]*w
        # py[:,:,1]=py[:,:,1]*h
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
        
        # cond_pred = []
        # for m in cond_ins_mask_t:
        #     t = cv2.resize(m, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
        #     #t = (t > self.threshold).astype(np.uint8)
        #     cond_pred.append(t)
        
        # #rles_cond = snake_eval_utils.binary_mask_to_rle(cond_pred)
        # image=Image.fromarray(batch['meta']['orig_img'].detach().cpu().numpy()[0])
        
        # dir="visual_pic/{}".format(self.i)
        # #visualize_contour(dir,output,batch)

        # image=Image.open(batch['meta']['path'][0])
        # dir="visual_pic/{}".format(self.i)
        # if os.path.exists(dir):
        #     shutil.rmtree(dir)
        # os.mkdir(dir)
        # shutil.copy(batch['meta']['path'][0],dir)
        # for i in range(len(py)):
        #     draw = ImageDraw.Draw(image)
        #     tmp=[]
        #     for j in range(len(py[i])):
        #         tmp.append((py[i][j][0],py[i][j][1]))
        #     draw.polygon(tmp,fill=None,outline='red')
        #     try:
        #         image.save(dir+"/poly_test{}_{}_{}.jpg".format(i,score[i],self.coco.cats[self.contiguous_category_id_to_json_id[label[i]]]['supercategory']))
        #     except:
        #         print('wrong')
        # for i in range(len(cond_pred)):
        #     #image=Image.open(batch['meta']['path'][0])
        #     src = cv2.imread(batch['meta']['path'][0])
        #     cond_pred[i] = np.uint8(cond_pred[i] * 255)
        #     mask_image = cv2.applyColorMap(cond_pred[i], cv2.COLORMAP_JET)
        #     #mask_image = Image.fromarray(cond_pred[i] * 255)
        #     #image.paste(mask_image, (0, 0),mask_image)
        #     superimposed_img = mask_image * 0.5 + src
        #     cv2.imwrite((dir+"/cond_test{}_{}_{}.jpg".format(i,score[i],self.coco.cats[self.contiguous_category_id_to_json_id[label[i]]]['supercategory'])),superimposed_img)
        #     #image.save(dir+"/cond_test{}_{}_{}.jpg".format(i,score[i],self.coco.cats[self.contiguous_category_id_to_json_id[label[i]]]['supercategory']))
        # self.i+=1

        coco_dets = []
        
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}


class DetectionEvaluator:
    def __init__(self, result_dir, logg):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        # box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach() * snake_config.down_ratio
        # print(py.shape)
        # input()
        if len(py) == 0:
            return 
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return
        # print(box.shape, box.min(), box.max(), scale, snake_config.down_ratio)
        # input()
        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        image=Image.fromarray(batch['meta']['orig_img'].detach().cpu().numpy())
        for i in range(len(py)):
            draw = ImageDraw.Draw(image)
            draw.polygon(py[i],fill=None,outline='red')
            image.save("poly_test{}".format(i))

        coco_dets = []
        for i in range(len(label)):
            box_ = data_utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'bbox': box_,
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}
        

Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
