from lib.core.api.classifier import Shufflenet
import numpy as np
import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
from lib.dataset.augmentor.augmentation import CenterCrop






def eval(models):


    face = Shufflenet(models)
    center_crop = CenterCrop(target_size=224, resize_size=256)

    with open('val.txt','r') as f:
        val_lines=f.readlines()

    top1_right=0
    top5_right=0
    total=0



    for one_line in tqdm(val_lines):
        line = one_line.rstrip()

        _img_path = line.rsplit('|', 1)[0]
        _label = int(line.rsplit('|', 1)[-1])

        image = cv2.imread(_img_path, cv2.IMREAD_COLOR)
        if cfg.DATA.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop(image)
        input=np.expand_dims(image,axis=0)
        label = np.array(_label)

        res=face.run(input)

        res=np.array(res[0])

        sorted_id=np.argsort(-res)



        if sorted_id[0]==label:
            top1_right+=1

        if np.sum(sorted_id[0:5]==label)==1:
            top5_right+=1


        total+=1


    print('top1 acc:%f'%(top1_right/total))
    print('top5 acc:%f'%(top5_right/total))

if __name__=='__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, default='the tf model', help="the tensorflow model")
    args = ap.parse_args()

    pb = args.model

    eval(pb)

