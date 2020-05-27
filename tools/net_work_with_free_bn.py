#-*-coding:utf-8-*-

import sys
sys.path.append('.')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import cv2


from train_config import config as cfg


from lib.core.model.net.shufflenet.shufflenetv2plus import ShufflenetV2Plus
from lib.core.model.net.shufflenet.shufflenetv2 import ShufflenetV2
from lib.core.model.net.shufflenet.shufflenetv2_5x5 import ShuffleNetV2_5x5

from lib.helper.logger import logger

from lib.core.base_trainer.metric import Metric



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", help="the trained ckpt file",
                    type=str)
args = parser.parse_args()
pretrained_model=args.pretrained_model



saved_file='./model/shufflenet_deploy.ckpt'
cfg.MODEL.deployee=True
if cfg.MODEL.deployee:
    cfg.TRAIN.batch_size = 1
    cfg.TRAIN.lock_basenet_bn=True

class trainner():
    def __init__(self):
        # self.train_ds=DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,True)
        # self.val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)


        self.inputs=[]
        self.outputs=[]
        self.val_outputs=[]
        self.ite_num=1



        self._graph=tf.Graph()

        self.summaries = []

        self.ema_weights = False

        self.metric=Metric(cfg.TRAIN.batch_size)

        self.train_dict={}
    def get_opt(self):

        with self._graph.as_default():
            ##set the opt there
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)

            # Decay the learning rate

            if cfg.TRAIN.lr_decay == 'cos':
                lr = tf.train.cosine_decay(
                    learning_rate=0.001, global_step=global_step, decay_steps=cfg.TRAIN.lr_decay_every_step[-1])
            else:
                lr = tf.train.piecewise_constant(global_step,
                                                 cfg.TRAIN.lr_decay_every_step,
                                                 cfg.TRAIN.lr_value_every_step
                                                 )
            if cfg.TRAIN.opt=='Adam':
                opt = tf.train.AdamOptimizer(lr)
            else:
                opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)

            if cfg.TRAIN.mix_precision:
                opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

            return opt,lr,global_step

    def load_weight(self):

        with self._graph.as_default():

            if 1:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
                print(variables_restore)

                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self._sess, pretrained_model)


    def add_summary(self, event):
        self.summaries.append(event)




    def tower_loss(self, scope, images, labels, training):
        """Calculate the total loss on a single tower running the model.

        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].

        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.


        if 'ShuffleNetV2_Plus' ==cfg.MODEL.net_structure:
            net = ShufflenetV2Plus
        elif 'ShuffleNetV2' ==cfg.MODEL.net_structure:
            net = ShufflenetV2
        elif 'ShuffleNetV2_5x5' == cfg.MODEL.net_structure:
            net = ShuffleNetV2_5x5
        else:
            raise NotImplementedError

        logits = net(images,False,include_head=True)

        mask=labels>=0

        labels = labels[mask]
        logits= logits[mask]
        onehot_labels=tf.one_hot(labels,depth=cfg.MODEL.cls)
        cls_loss=slim.losses.softmax_cross_entropy(logits=logits,onehot_labels=onehot_labels,label_smoothing=0.1)


        predicts = tf.nn.softmax(logits=logits)

        correct_prediction = tf.equal(tf.argmax(predicts, 1), labels)
        top1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        top5_correct_prediction = tf.nn.in_top_k(logits, labels,  k = 5)
        top5_accuracy = tf.reduce_mean(tf.cast(top5_correct_prediction, "float"), name="top5_accuracy")

        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        ### make loss and acc
        return cls_loss,top1_accuracy,top5_accuracy, l2_loss

    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                try:
                    expanded_g = tf.expand_dims(g, 0)
                except:
                    print(_)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def build(self):
        with self._graph.as_default(), tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.

            # Create an optimizer that performs gradient descent.
            opt, lr, global_step = self.get_opt()

            training = tf.placeholder(tf.bool, name="training_flag")

            images_place_holder_list = []
            labels_place_holder_list = []
            # Create an optimizer that performs gradient descent.

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                                images_ = tf.placeholder(tf.float32, [None, cfg.MODEL.hin, cfg.MODEL.win, 3],
                                                         name="images")
                                labels_ = tf.placeholder(tf.int64, [None], name="labels")
                                images_place_holder_list.append(images_)
                                labels_place_holder_list.append(labels_)


                                cls_loss,top1_accuracy,top5_accuracy, l2_loss = self.tower_loss(
                                    scope, images_, labels_, training)

                                ##use muti gpu ,large batch
                                if i == cfg.TRAIN.num_gpu - 1:
                                    total_loss = tf.add_n([cls_loss, l2_loss])
                                else:
                                    total_loss = tf.add_n([cls_loss])


                                # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()

                                ##when use batchnorm, updates operations only from the
                                ## final tower. Ideally, we should grab the updates from all towers
                                # but these stats accumulate extremely fast so we can ignore the
                                #  other stats from the other towers without significant detriment.
                                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                                # Retain the summaries from the final tower.
                                # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                                summaries = tf.get_collection('%smutiloss' % scope, scope)
                                # Calculate the gradients for the batch of data on this CIFAR tower.
                                grads = opt.compute_gradients(total_loss)

                                # Keep track of the gradients across all towers.
                                tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            self.add_summary(tf.summary.scalar('learning_rate', lr))
            self.add_summary(tf.summary.scalar('loss', cls_loss))
            self.add_summary(tf.summary.scalar('acctop1', top1_accuracy))
            self.add_summary(tf.summary.scalar('acctop5', top5_accuracy))
            self.add_summary(tf.summary.scalar('l2_loss', l2_loss))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            if self.ema_weights:
                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(
                    0.9, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # Group all updates to into a single train op.
                train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            else:
                train_op = tf.group(apply_gradient_op, *bn_update_ops)

            self.inputs = [images_place_holder_list, labels_place_holder_list, training]
            self.outputs = [train_op, total_loss,  cls_loss,top1_accuracy,top5_accuracy, l2_loss, lr]
            self.val_outputs = [total_loss,  cls_loss,top1_accuracy,top5_accuracy, l2_loss, lr]
            # Create a saver.

            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()


            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            #tf_config.gpu_options.allow_growth = True
            tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            tf_config.intra_op_parallelism_threads = 18

            self._sess = tf.Session(config=tf_config)
            self._sess.run(init)







    def loop(self,):

        self.build()
        self.load_weight()


        with self._graph.as_default():
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


            logger.info('A tmp model  saved as %s \n' % saved_file)
            self.saver.save(self._sess, save_path=saved_file)


    def train(self):
        self.loop()






tmp_trainer=trainner()

tmp_trainer.train()


