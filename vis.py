from lib.dataset.dataietr import DataIter
from train_config import config
from lib.core.api.classifier import Shufflenet
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from train_config import config as cfg
cfg.TRAIN.batch_size=1

#val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)

face=Shufflenet('./model/shufflenet.pb')


#
# for one_ele,_, in val_ds:
#     if 1:
#         img_show=np.array(one_ele)
#         res=face.run(one_ele)
#         #print(res)
#         res=np.array(res)
#
#
#         img_show=img_show.astype(np.uint8)[0]
#
#
#         cv2.imshow('tmp',img_show)
#         cv2.waitKey(0)



img=cv2.imread('test.jpg')
img=cv2.resize(img,(224,224))
img=np.expand_dims(img,0)

img=np.array(img,dtype=np.float32)

res=face.run(img)

print(np.argmax(res))
#         #print(res)