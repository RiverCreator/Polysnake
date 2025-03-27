import torch
from .utils import masks2patch
from detectron2.layers import cat
from lib.csrc.roi_align_layer.roi_align import ROIAlign

class GT_infomation:
    def __init__(self,mask_size_assemble, mask_size, patch_size, scale, dct_encoding, patch_dct_encoding):
        self.mask_size_assemble = mask_size_assemble
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.scale = scale
        self.dct_encoding = dct_encoding
        self.patch_dct_encoding =  patch_dct_encoding
        #self.gt_pooler = ROIAlign # => (28,28)
        
    def crop_and_resize(self, gt_masks, rois, mask_sizes):
        #将gt_masks按预测的box裁剪后，使用roi align变为mask_sizes
        device = gt_masks.device
        batch_inds = torch.arange(len(rois), device=device).to(dtype=rois.dtype)[:, None]
        rois = torch.cat([batch_inds, rois[:,1:]], dim=1)  # Nx5
        align_class = ROIAlign((mask_sizes, mask_sizes))
        output = align_class(gt_masks[:,None,:,:], rois).squeeze(1)
        output = output >= 0.5
        return output
    
    def get_gt_mask(self,per_ins_cmask , rois):
        #针对单个实例预测 在ICD中，
        #gt_masks = []
        gt_classes = []
        gt_masks_coarse = []
        
        # for i in range(len(output['mask_preds'])):
        #     pred_masks = output['mask_preds'][i]
        #pred_masks = pred_masks[torch.arange(pred_masks.shape[0]),batch['ct_cls'][batch['ct_01'].byte()]]
        gt_masks_coarse = self.crop_and_resize(per_ins_cmask, rois, self.mask_size) # 获得对应rois的gt mask，resize为mask_size
        
        gt_masks = self.crop_and_resize(per_ins_cmask, rois, self.mask_size_assemble) #【n，mask_size_assemble，mask_size_assemble】
        gt_masks_per_image = masks2patch(gt_masks,self.scale,self.patch_size,self.mask_size_assemble) # [B*num_patch,patch_size,patch_size] num_patch即分成了多少个patch
        
        gt_masks = self.patch_dct_encoding.encode(gt_masks) # 将gt_masks encode为 dct编码的
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        
        gt_masks_coarse = self.dct_encoding.encode(gt_masks_coarse).to(dtype=torch.float32) # gt_masks_coarse也要进行encoding
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks) #gt_bfg是将dct_encoding后的bfg进行处理，0是前景 1是边缘 2是背景 

        return gt_masks,gt_masks_coarse,gt_bfg
        
        for instances_per_image in instances:

            if len(instances_per_image) == 0:
                continue

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size)
            gt_masks_coarse.append(gt_masks_per_image) #source mask concat

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size_assemble)
            # divided masks into scalexscale patch,patch size=8
            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append(gt_masks_per_image) #patch concat

            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)  # [N, dct_v_dim]
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        gt_classes = cat(gt_classes, dim=0) #[N_instanc]
        gt_masks_coarse = cat(gt_masks_coarse,dim=0)
        gt_masks_coarse = self.dct_encoding.encode(gt_masks_coarse).to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_classes,gt_masks_coarse,gt_bfg

    #TODO inference 还需要修改
    def get_gt_mask_inference(self,instances,pred_mask_logits):
        gt_masks = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if instances_per_image.has("gt_masks"):
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.pred_boxes.tensor, self.mask_size_assemble)
            else:
                #print("gt_mask is empty")
                shape = instances_per_image.pred_boxes.tensor.shape[0]
                device = instances_per_image.pred_boxes.tensor.device
                gt_masks_per_image = torch.zeros((shape,self.mask_size_assemble,self.mask_size_assemble),dtype=torch.bool).to(device)

            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append((gt_masks_per_image))

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)
        gt_masks = gt_masks.to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_bfg

    def get_gt_bfg(self, gt_masks):
        gt_bfg = gt_masks[:, 0].clone()
        gt_bfg[(gt_bfg > 0) & (gt_bfg < self.patch_size)] = 1.
        gt_bfg[gt_bfg == self.patch_size] = 2
        gt_bfg = gt_bfg.to(dtype=torch.int64)
        gt_masks = gt_masks[gt_bfg == 1, :]
        return gt_masks, gt_bfg
    
    def get_gt_classes(self, instances):
        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0)  # [N_instance]
        return gt_classes
