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
import random
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
        self.vis_imgid = {3204,25912,28114,28123,43203,66204,68101,68012}
        self.kins_vis_imgid = {4245,4246,4247,4248,4249,4250,4251,4252,4253,4254,4255,4256,4257,4258,4259,4260,4275,4276,4277,4278,4279,4280,4281,4282,4283,4284,4284,4285,4286,4289}
        
    def vis_data(self, mask, batch, imgid, img_name):
        import numpy as np
        from PIL import Image
        image_path = batch['meta']['path'][0]  # 读取原图路径
        H, W = mask.shape[1], mask.shape[2]
        # 加载原图
        original_img = Image.open(image_path).convert("RGBA")
        N = mask.shape[0]
        # 遍历每个实例
        orig_w, orig_h = original_img.size  # 获取原图尺寸

        # 遍历每个实例
        for i in range(N):
            # 获取当前实例的 mask
            mask_np = mask[i].numpy().astype(np.uint8) * 255  # 转换为 uint8 格式 (0, 255)

            # 创建一个空的 RGBA mask 图层
            mask_img = Image.new("L", (W, H), 0)  # 纯黑色 (L模式：单通道灰度)
            mask_img.paste(Image.fromarray(mask_np, mode="L"))  # 只在 mask 位置填充白色 (255)

            # 调整 mask 大小，使其匹配原图
            mask_img = mask_img.resize((orig_w, orig_h))

            # 创建带透明度的颜色层
            mask_colored = Image.new("RGBA", original_img.size, (255, 0, 0, 100))  # 半透明红色
            mask_colored.putalpha(mask_img)  # 透明度由 mask 控制

            # 叠加 mask 到原图
            blended_img = Image.alpha_composite(original_img, mask_colored)
            img_name = img_name + str(i)
            # 保存结果
            blended_img.save('visb/mask{}_{}.png'.format(imgid,img_name), format='PNG')
            
        # # 假设您的掩码张量存储在变量 mask 中，形状为 (1, 1, 168, 128)
        # # 这里我们使用随机数据作为示例
        # #mask = np.random.randn(1, 1, 168, 128)

        # # 去除多余的维度，得到形状为 (168, 128) 的二维数组
        # mask = np.squeeze(mask)

        # #mask_normalized = torch.sigmoid(mask)

        # # 将归一化后的张量值缩放到 0-255 范围，并转换为无符号8位整数类型
        # mask_scaled = (mask * 255).byte()

        # # 将张量转换为 NumPy 数组
        # mask_np = mask_scaled.cpu().numpy()

        # # 将 NumPy 数组转换为 PIL 图像
        # mask_image = Image.fromarray(mask_np)

        # # 保存为 JPEG 格式的图像
        # mask_image.save('visb/mask{}_{}.jpg'.format(imgid,img_name), format='JPEG')
        self.i += 1
    
    
    def vis_poly(self, py, label, batch, img, type = "d2sa"):
        #visualize_contour(dir,output,batch)
        if(type == "d2sa"):
            image = Image.open("/data0/river/Polysnake/data/d2sa/images/{}".format(img['file_name'])).convert('RGBA')
        elif(type == "kins"):
            image = Image.open("/data0/river/Polysnake/data/kitti/testing/image_2/{}".format(img['file_name'])).convert('RGBA')
        elif(type == "cocoa"):
            image = Image.open("/data0/river/Polysnake/data/cocoa/val2014/{}".format(img['file_name'])).convert('RGBA')
        #image=Image.fromarray(batch['meta']['orig_img'].detach().cpu().numpy()[0])
        dir="vis_{}717/{}".format(type,self.i)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        shutil.copy(batch['meta']['path'][0],dir)
        for i in range(len(py)):
            image2 = Image.new("RGBA", (img['width'], img['height']))
            draw = ImageDraw.Draw(image2)
            tmp=[]
            for j in range(len(py[i])):
                tmp.append((py[i][j][0],py[i][j][1]))
            
            polygon_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)
            draw.polygon(tmp, fill= polygon_color, outline=polygon_color)
            blend = Image.alpha_composite(image, image2)
            try:
                blend.save(dir+"/poly_test_{}_{}_{}.png".format(i,int(batch['meta']['img_id'][0]),self.coco.cats[self.contiguous_category_id_to_json_id[label[i]]]['supercategory']))
            except:
                print('wrong')
        self.i+=1
    
    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        i_gt_py = batch['i_gt_py'][0].detach().cpu().numpy() * snake_config.down_ratio
        # cond_ins_mask=output['cond_predict_val'].sigmoid()
        # cond_ins_mask_t = np.asarray(cond_ins_mask.cpu())
        #py_init=output['poly_init_infer'][output['idx']][output['nms_keep']].cpu().numpy()
        #py=py_init
        if len(py) == 0:
            return
        
        img_id = int(batch['meta']['img_id'][0])
        # self.vis_data(output['amodal_preds'][-1][0][label[0]],img_id)
        # self.vis_data(output['per_ins_cmask'][0][0],img_id,)
            # self.vis_data(batch['per_ins_cmask'][0], batch, img_id, "amodal")
            # self.vis_data(batch['per_vis_cmask'][0], batch, img_id,"vis")
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy() #d2sa:(1952,1504)
        
        h, w = batch['inp'].size(2), batch['inp'].size(3)
        # py[:,:,0]=py[:,:,0]*w
        # py[:,:,1]=py[:,:,1]*h
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        
        img = self.coco.loadImgs(img_id)[0]
        #self.vis_poly(py,label,batch,img['file_name'])
        ori_h, ori_w = img['height'], img['width']
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
        if(cfg.need_vis):
            if("d2sa" in cfg.model):
                if(img_id in self.vis_imgid):
                    self.vis_poly(py,label,batch,img,type="d2sa")
            elif("kins" in cfg.model):
                if(img_id in self.kins_vis_imgid):
                    self.vis_poly(py,label,batch,img,type="kins")
            elif("cocoa" in cfg.model):
                self.vis_poly(py,label,batch,img,type="cocoa")
        # if('001119' in img['file_name'] or '68024' in img['file_name']):
        #     self.vis_poly(py,label,batch,img)
        #self.vis_poly(py,label,batch)

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
