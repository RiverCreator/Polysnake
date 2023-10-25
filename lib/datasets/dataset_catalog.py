from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'CocoTrain': {
            'id': 'coco',
            'data_root': 'data/coco/train2017',
            'ann_file': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'CocoVal': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'test'
        },
        'CocoMini': {
            'id': 'coco',
            'data_root': 'data/coco/val2017',
            'ann_file': 'data/coco/annotations/instances_val2017.json',
            'split': 'mini'
        },
        'CocoTest': {
            'id': 'coco_test',
            'data_root': 'data/coco/test2017',
            'ann_file': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'CityscapesTrain': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'CityscapesVal': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'CityscapesCocoVal': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'CityCocoBox': {
            'id': 'cityscapes_coco',
            'data_root': 'data/cityscapes/leftImg8bit/val',
            'ann_file': 'data/cityscapes/coco_ann/instance_box_val.json',
            'split': 'val'
        },
        'CityscapesMini': {
            'id': 'cityscapes',
            'data_root': 'data/cityscapes/leftImg8bit',
            'ann_file': 'data/cityscapes/annotations/val',
            'split': 'mini'
        },
        'CityscapesTest': {
            'id': 'cityscapes_test',
            'data_root': 'data/cityscapes/leftImg8bit/test'
        },
        'SbdTrain': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'SbdVal': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'SbdMini': {
            'id': 'sbd',
            'data_root': 'data/sbd/img',
            'ann_file': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'mini'
        },
        'VocVal': {
            'id': 'voc',
            'data_root': 'data/voc/JPEGImages',
            'ann_file': 'data/voc/annotations/voc_val_instance.json',
            'split': 'val'
        },
        'KinsTrain': {
            'id': 'kins',
            'data_root': 'data/kitti/training/image_2',
            'ann_file': 'data/kitti/training/instances_train.json',
            'split': 'train'
        },
        'KinsVal': {
            'id': 'kins',
            'data_root': 'data/kitti/testing/image_2',
            'ann_file': 'data/kitti/testing/instances_val.json',
            'split': 'val'
        },
        'KinsMini': {
            'id': 'kins',
            'data_root': 'data/kitti/testing/image_2',
            'ann_file': 'data/kitti/testing/instances_val.json',
            'split': 'mini'
        },
        'CocoaTrain': {
            'id': 'cocoa',
            'data_root': 'data/cocoa/train2014',
            'ann_file': 'data/cocoa/amodal_cls_annotations/COCO_amodal_train2014_with_classes_amodal.json',
            'split': 'train'
        },
        'CocoaVal': {
            'id': 'cocoa',
            'data_root': 'data/cocoa/val2014',
            'ann_file': 'data/cocoa/amodal_cls_annotations/COCO_amodal_val2014_with_classes.json',
            'split': 'val'
        },
        'CocoaMini': {
            'id': 'cocoa',
            'data_root': 'data/cocoa/val2014',
            'ann_file': 'data/cocoa/amodal_cls_annotations/COCO_amodal_val2014_with_classes.json',
            'split': 'mini'
        },
        'CocoaTest' : {
             'id': 'coco_test',
            'data_root': 'data/cocoa/val2014',
            'ann_file': 'data/cocoa/amodal_annotations/COCO_amodal_test2014.json',
            'split': 'test'   
        },
        'D2saTrain':{
            'id': 'd2sa',
            'data_root': 'data/d2sa/images',
            'ann_file': ['data/d2sa/annotations/D2S_amodal_augmented_amodal.json','data/d2sa/annotations/D2S_amodal_training_rot0.json'],
            'split': 'train' 
        },
        'D2saVal':{
            'id': 'd2sa',
            'data_root': 'data/d2sa/images',
            'ann_file': 'data/d2sa/annotations/D2S_amodal_validation.json',
            'split': 'mini' 
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

