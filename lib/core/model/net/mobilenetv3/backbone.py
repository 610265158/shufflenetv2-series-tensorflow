import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenetv3 import mobilnet_v3
from lib.core.model.net.mobilenet.mobilenet import training_scope
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish

def mobilenetv3_large_detection(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):

        _, endpoints = mobilnet_v3.large(image,
                                        depth_multiplier=0.75,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)

        for k,v in endpoints.items():
            print('mobile backbone output:',k,v)

        extern_conv = slim.conv2d(_,
                                  480,
                                  [1, 1],
                                  stride=1,
                                  padding='SAME',
                                  activation_fn=hard_swish,
                                  scope='extern1')

        print(extern_conv)
        mobilebet_fms = [endpoints['layer_5/expansion_output'],
                         endpoints['layer_7/expansion_output'],
                         endpoints['layer_13/output'],
                         extern_conv]

    return mobilebet_fms


def mobilenetv3_small_minimalistic(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):
            
        final_feature, endpoints = mobilnet_v3.small_minimalistic(image,
                                        depth_multiplier=1.0,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)

        x = tf.reduce_mean(final_feature, axis=[1, 2], keep_dims=True)

        x = slim.dropout(x, 0.5)

        x = slim.conv2d(x,
                        num_outputs=cfg.MODEL.cls,
                        kernel_size=[1, 1],
                        stride=1,
                        padding='VALID',
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='classifier/0')


    x = tf.squeeze(x, axis=1)
    x = tf.squeeze(x, axis=1)
    x = tf.identity(x, name='cls_output')
    return x
