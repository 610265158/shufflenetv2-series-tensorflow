import torch
import tensorflow as tf
import numpy as np

from train_config import config as cfg

params_dict=np.load(cfg.MODEL.pretrained_model,allow_pickle=True).item()



y = np.random.rand(1,224,224,3).astype(np.float32)
filterx = params_dict['ShuffleNetV2_Plus/first_conv/0/weights:0']

filterx=np.array(filterx,dtype=np.float32)


kernel_size_effective = 3
pad_total = kernel_size_effective - 1
pad_beg = pad_total // 2
pad_end = pad_total - pad_beg
inputs = tf.pad(y,
                [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

a= tf.nn.conv2d(
    inputs,
    filterx, 2, padding='VALID')

with tf.Session() as sess:
  t = (sess.run(a))



x = torch.nn.Conv2d(3,16,3,stride=2,padding=1, bias = False)
filter = np.transpose(filterx, (3,2,0,1))
x.weight = torch.nn.Parameter(torch.from_numpy(filter))
z = np.transpose(y, (0,3,1,2))
l = x(torch.from_numpy(z))
l = l.detach().numpy()
l = np.transpose(l,(0,2,3,1))

print(t.shape)
print(l.shape)
print(np.abs(t-l).max())