#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import cv2


from train_config import config as cfg
from lib.dataset.dataietr import DataIter

from lib.core.model.net.shufflenet.shufflenetv2plus import ShufflenetV2Plus
from lib.core.model.net.shufflenet.shufflenetv2 import ShufflenetV2
from lib.core.model.net.shufflenet.shufflenetv2_5x5 import ShuffleNetV2_5x5

from lib.helper.logger import logger

from lib.core.base_trainer.metric import Metric

class trainner():
    def __init__(self):
        self.train_ds=DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,True)
        self.val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,False)


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

            if cfg.MODEL.continue_train:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
                print(variables_restore)

                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self._sess, cfg.MODEL.pretrained_model)

            elif 'npy' in cfg.MODEL.pretrained_model:


                params_dict=np.load(cfg.MODEL.pretrained_model,allow_pickle=True).item()

                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)




                ##filter
                if cfg.MODEL.cls != 1000:
                    variables_restore = [x for x in variables_restore if 'classifier' not in x.name]

                print(variables_restore)
                for i,variables in enumerate(variables_restore):

                    logger.info('assign %s with np data'%(variables.name) )

                    self._sess.run(variables.assign( params_dict[variables.name]))

            elif cfg.MODEL.pretrained_model is not None :
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)



                if cfg.MODEL.cls != 1000:
                    variables_restore=[x for x in variables_restore if 'classifier' not in x.name]
                print(variables_restore)
                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self._sess, cfg.MODEL.pretrained_model)



            else:
                variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=cfg.MODEL.net_structure)
                print(variables_restore)
                logger.info('no pretrained model, train from sctrach')
                # Build an initialization operation to run below.

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

        logits = net(images,training,include_head=True)

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

            tmp_model_name = cfg.MODEL.model_path + '/cls_for_convert.ckpt'
            logger.info('A tmp model  saved as %s \n' % tmp_model_name)
            self.saver.save(self._sess, save_path=tmp_model_name)

            # Build the summary operation from the last tower summaries.
            self.summary_op = tf.summary.merge(self.summaries)

            self.summary_writer = tf.summary.FileWriter(cfg.MODEL.model_path, self._sess.graph)

        min_loss_control=1000.
        for epoch in range(cfg.TRAIN.epoch):
            self._train(epoch)
            val_loss=self._val(epoch)
            logger.info('**************'
                       'val_loss %f '%(val_loss))

            #tmp_model_name=cfg.MODEL.model_path + \
            #               'epoch_' + str(epoch ) + \
            #               'L2_' + str(cfg.TRAIN.weight_decay_factor) + \
            #               '.ckpt'
            #logger.info('save model as %s \n'%tmp_model_name)
            #self.saver.save(self.sess, save_path=tmp_model_name)

            if 1:
                min_loss_control=val_loss
                low_loss_model_name = cfg.MODEL.model_path + \
                                 'epoch_' + str(epoch) + \
                                 'L2_' + str(cfg.TRAIN.weight_decay_factor)  + '.ckpt'
                logger.info('A new low loss model  saved as %s \n' % low_loss_model_name)
                self.saver.save(self._sess, save_path=low_loss_model_name)

        self._sess.close()


    def _train(self,_epoch):
        for step in range(cfg.TRAIN.iter_num_per_epoch):
            self.ite_num += 1
            start_time = time.time()

            example_images, example_labels = next(self.train_ds)

            ########show_flag check the data
            if cfg.TRAIN.vis:
                for i in range(cfg.TRAIN.batch_size):
                    example_image = example_images[i, :, :, :]
                    example_label = example_labels[i]


                    print(example_label)
                    cv2.namedWindow('img', 0)
                    cv2.imshow('img', example_image.astype(np.uint8))
                    cv2.waitKey(0)

            fetch_duration = time.time() - start_time


            for n in range(cfg.TRAIN.num_gpu):
                self.train_dict[self.inputs[0][n]] = example_images[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size, :,:,:]
                self.train_dict[self.inputs[1][n]] = example_labels[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

            self.train_dict[self.inputs[2]] = True
            _, total_loss_value, loss_value, top1_acc_value, top5_acc_value, l2_loss_value, learn_rate, = \
                self._sess.run([*self.outputs],
                         feed_dict=self.train_dict)



            duration = time.time() - start_time
            run_duration = duration - fetch_duration
            if self.ite_num % cfg.TRAIN.log_interval == 0:
                num_examples_per_step = cfg.TRAIN.batch_size * cfg.TRAIN.num_gpu
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / cfg.TRAIN.num_gpu

                format_str = ('epoch %d: iter %d, '
                              'total_loss=%.6f '
                              'loss=%.6f '
                              'top1 acc=%.6f '
                              'top5 acc=%.6f '
                              'l2_loss=%.6f '
                              'learn_rate =%e '
                              '(%.1f examples/sec; %.3f sec/batch) '
                              'fetch data time = %.6f'
                              'run time = %.6f')
                logger.info(format_str % (_epoch,
                                          self.ite_num,
                                          total_loss_value,
                                          loss_value,
                                          top1_acc_value,
                                          top5_acc_value,
                                          l2_loss_value,
                                          learn_rate,
                                          examples_per_sec,
                                          sec_per_batch,
                                          fetch_duration,
                                          run_duration))

            # if self.ite_num % 100 == 0:
            #     summary_str = self._sess.run(self.summary_op, feed_dict=self.train_dict)
            #     self.summary_writer.add_summary(summary_str, self.ite_num)

    def _val(self,_epoch):


        all_total_loss=0
        for step in range(cfg.TRAIN.val_iter):

            example_images, example_labels = next(self.val_ds)  # 在会话中取出image和label

            feed_dict = {}
            for n in range(cfg.TRAIN.num_gpu):
                feed_dict[self.inputs[0][n]] = example_images[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size, :,:,:]
                feed_dict[self.inputs[1][n]] = example_labels[n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
            feed_dict[self.inputs[2]] = False
            total_loss_value, loss_value, top1_acc_value, top5_acc_value, l2_loss_value, learn_rate = \
                self._sess.run([*self.val_outputs],
                              feed_dict=feed_dict)

            all_total_loss+=total_loss_value-l2_loss_value
            self.metric.update(top1_acc_value,top5_acc_value)
        self.metric.report()
        return all_total_loss/cfg.TRAIN.val_iter

    def train(self):
        self.loop()









