

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()
#### below are params for dataiter
config.TRAIN.process_num = 5
config.TRAIN.prefetch_size = 50
############


config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 128
config.TRAIN.save_interval = 5000               ##no use, we save the model evry epoch
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.epoch = 300                       #### no actual meaning, just keep training,
config.TRAIN.train_set_size=1281167              ###########u need be sure
config.TRAIN.val_set_size=50000                ###50562

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.001,0.01,0.01,0.001,0.0001,0.00001]          ####lr policy
config.TRAIN.lr_decay_every_step = [300,500,600000,800000,1000000]
config.TRAIN.lr_decay='cos'

config.TRAIN.weight_decay_factor = 4.e-5                                    ####l2
config.TRAIN.train_val_ratio= 0.9                                           ### nouse
config.TRAIN.vis=False
#### if to check the training data
config.TRAIN.mix_precision=True                                            ##use mix precision to speedup, tf1.14 at least
config.TRAIN.opt='Adam'                                                     ##Adam or SGD， sgd is more stable for resnet

config.MODEL = edict()
config.MODEL.continue_train=False                                           ##recover from a model completly
config.MODEL.model_path = './model/'                                        ## save directory
config.MODEL.hin = 224                                                      # input size during training , 128,160,   depends on
config.MODEL.win = 224
config.MODEL.cls=1000

config.MODEL.net_structure='ShuffleNetV2_Plus'
config.MODEL.pretrained_model='ShuffleNetV2+.Small.pth.tar.npy'                    ##according to your model,
config.MODEL.size='Small'     ##Small Medium Large   for v2+
                                ##0.5x, 1.0x 1.5x 2.0x   for v2


config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.rgb=True
############ the model is trained with BGR mode



config.MODEL.deployee=False