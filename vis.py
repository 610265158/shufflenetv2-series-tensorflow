import numpy as np
import cv2

from lib.core.api.classifier import Shufflenet


import argparse
from lib.dataset.augmentor.augmentation import CenterCrop


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, default='test,jpg', help="the image used to predict")
ap.add_argument("--model", required=True, default='ShuffleNetV2_Plus.pb', help="the model used")
args = ap.parse_args()
input_file=args.input
model_file=args.model

classifier=Shufflenet(model_file)

img=cv2.imread(input_file)
img = CenterCrop(target_size=224, resize_size=256)

img=np.expand_dims(img,0)

img=np.array(img,dtype=np.float32)

res=classifier.run(img)
res=np.array(res[0])
sorted_id=np.argsort(-res)

print('class id is ',sorted_id)
