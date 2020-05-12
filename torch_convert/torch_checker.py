import sys
sys.path.append('.')

import torch
from torch import nn
import numpy as np
import cv2
from torch_convert.network import ShuffleNetV2_Plus

architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
model = ShuffleNetV2_Plus(architecture=architecture,model_size='Small')
model = nn.DataParallel(model)

#
#
# test_data = torch.rand(5, 3, 224, 224)
# test_outputs = model(test_data)
# print(test_outputs.size())

params=torch.load('ShuffleNetV2+.Small.pth.tar', map_location=torch.device('cpu'))





params_dict=params['state_dict']
model.load_state_dict(params_dict)


img=cv2.imread('test.jpg')
img=cv2.resize(img,(224,224))
img=np.transpose(img,axes=[2,0,1])
img=np.expand_dims(img,0)


img=np.array(img,dtype=np.float32)
input=torch.from_numpy(img)
model=model.to('cpu')
model.eval()
out=model(input)

# print(out)