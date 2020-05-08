


import os
import random
import cv2
import numpy as np
from functools import partial
import traceback
import copy

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, MultiProcessPrefetchData


from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                        Affine_aug,\
                                        Mirror,\
                                        Padding_aug,\
                                        Img_dropout,\
                                        RandomResizedCrop,\
                                        OpencvResize,\
                                        CenterCrop

from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter
from lib.dataset.headpose import get_head_pose
from train_config import config as cfg


class data_info(object):
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()

        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('|',1)[0]
            _label=line.rsplit('|',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas


class DataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True):

        self.shuffle=True
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size
        if not training_flag:
            self.process_num=1
        self.generator = ImageNetDataIter(img_root_path, ann_file, self.training_flag)

        self.ds=self.build_iter()


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self):

        map_func=partial(self._map_func,is_training=self.training_flag)
        ds = DataFromGenerator(self.generator)
        ds = BatchData(ds, self.num_gpu *  self.batch_size)
        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.ds)


    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class ImageNetDataIter():
    def __init__(self, img_root_path='', ann_file=None, training_flag=True,shuffle=True):


        self.training_flag = training_flag
        self.shuffle = shuffle

        self.color_augmentor = ColorDistort()
        self.random_crop_resize=RandomResizedCrop(size=(cfg.MODEL.hin,cfg.MODEL.win))
        self.center_crop=CenterCrop(target_size=224,resize_size=256)

        self.lst = self.parse_file(img_root_path, ann_file)




    def __iter__(self):
        idxs = np.arange(len(self.lst))

        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self._map_func(self.lst[k], self.training_flag)


    def parse_file(self,im_root_path,ann_file):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        fname, ann = dp
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = np.array(ann)

        if is_training:

            image=self.random_crop_resize(image)

            if random.uniform(0, 1) > 0.5:
                image, _ = Mirror(image, label=None, symmetry=None)
            if random.uniform(0, 1) > 0.0:
                angle = random.uniform(-15, 15)
                image, _ = Rotate_aug(image, label=None, angle=angle)

            if random.uniform(0, 1) > 1.:
                strength = random.uniform(0, 50)
                image, _ = Affine_aug(image, strength=strength, label=None)

            if random.uniform(0, 1) > 0.5:
                image=self.color_augmentor(image)
            if random.uniform(0, 1) > 1.0:
                image=pixel_jitter(image,15)
            if random.uniform(0, 1) > 0.5:
                image = Img_dropout(image, 0.2)

        else:
            ###centercrop
            image = self.center_crop(image)


        label = label.astype(np.int64)
        image= image.astype(np.uint8)
        return image, label
