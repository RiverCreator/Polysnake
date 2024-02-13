# import torch
# import os
# from termcolor import colored
# model_dir='data/model/snake/kins_snake'
# model_name='best'
# if not os.path.exists(model_dir):
#     print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
# pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(model_name)))
# ap=pretrained_model['ap']
# print(colored('model ap is {}'.format(ap)))

##### convert dataset
# from pycocotools.coco import COCO
# import os
# def test():
#     ann_file='data/d2sa/annotations/D2S_amodal_validation.json'
#     coco=COCO(ann_file)
#     print(coco)
# test()
# def process_info(coco,img_id):
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anno = coco.loadAnns(ann_ids)
#     #path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
#     return anno, img_id
# ann1_file='data/cocoa/amodal_annotations/COCO_amodal_val2014.json'
# ann2_file='data/cocoa/amodal_cls_annotations/COCO_amodal_val2014_with_classes.json'
# ann3_file='data/cocoa/amodal_cls_annotations/COCO_amodal_train2014_with_classes.json'
# coco1=COCO(ann1_file)
# coco2=COCO(ann2_file)

# anns1=sorted(coco1.getImgIds())
# anns2=sorted(coco2.getImgIds())
# anno1,img_id1=process_info(coco1,anns1[0])
# anno2,img_id2=process_info(coco2,anns2[0])
# catagory_id=coco2.getCatIds()
# print("ok")

# import json
# cats=set()
# for idx in range(0,len(anns1)):
#     ann,img_id=process_info(coco1,anns1[idx])
#     for obj in ann[0]['regions']:
#         cats.add(obj['name'])
# print(cats)
# print(len(cats)) 
# train_class_dataset=json.load(open("data/cocoa/amodal_annotations/COCO_amodal_val2014.json", 'r'))  ## 1323张图片，
# val_class_dataset=json.load(open("data/cocoa/amodal_cls_annotations/COCO_amodal_val2014_with_classes.json", 'r'))  ##总共3799个实例
# for i in range(len(train_class_dataset)):


# # map_dict=dict()
# # for obj in val_class_dataset['categories']:
# #     map_dict[obj['name']]=obj['id']
# # print(map_dict)


# #### debug show pic

# from PIL import Image,ImageDraw
# image=Image.open(path)
# #newsize=(int(image.width/4),int(image.height/4))
# #image=image.resize(newsize)
# draw = ImageDraw.Draw(image)
# tmp=[]
# # for point in i_gt_pys[0]:
# #     tmp.append((point[0],point[1]))
# for idx in range(0,len(obj),2):
#     tmp.append((obj[idx],obj[idx+1]))
# draw.polygon(tmp,fill="red",outline="red")
# image.save("poly_test.png")

# # from PIL import Image,ImageDraw
# # image=Image.open(path)
# # print(image.height/4)


# # from PIL import Image
# # img=Image.fromarray(arr)
# # img.save("origin_img")
# from PIL import Image,ImageDraw
# import cv2
# img=Image.open('test.jpg')
# Draw=ImageDraw.Draw(img)
# xmin=20
# ymin=90
# xmax=50
# ymax=140

# for proposals in results['gt_bboxes_2d']:
#     xmin=int((proposals[0]+crop[0])/resize)
#     ymin=int((proposals[2]+crop[1])/resize)
#     xmax=int((proposals[1]+crop[0])/resize)
#     ymax=int((proposals[3]+crop[1])/resize)
