import sys
sys.path.append('.')
import os
import tensorflow as tf
from train_config import config as cfg




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", help="the trained file,  end with .ckpt",
                    type=str)
parser.add_argument("--net_structure", help="the net structure",
                    type=str)
parser.add_argument("--size", help="the trained file,  end with .ckpt",
                    type=float)



args = parser.parse_args()
pretrained_model=args.pretrained_model

cfg.MODEL.net_structure=args.net_structure
cfg.MODEL.size=args.size

print(pretrained_model)

command="python tools/net_work_with_free_bn.py --pretrained_model %s "%pretrained_model
os.system(command)
print('save ckpt with bn defaut False')




model_folder = cfg.MODEL.model_path
checkpoint = tf.train.get_checkpoint_state(model_folder)

##input_checkpoint
input_checkpoint = checkpoint.model_checkpoint_path
##input_graph
input_meta_graph = input_checkpoint + '.meta'

##output_node_names
output_node_names='tower_0/images,tower_0/cls_output'

#output_graph
output_graph='./model/shufflenet.pb'

print('excuted')

command="python tools/freeze.py --input_checkpoint %s --input_meta_graph %s --output_node_names %s --output_graph %s"\
%(input_checkpoint,input_meta_graph,output_node_names,output_graph)
os.system(command)
