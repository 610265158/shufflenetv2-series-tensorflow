#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import time

from lib.helper.init import init
from train_config import config
class Shufflenet():

    def __init__(self,pb):
        self.PADDING_FLAG=True
        self.ANCHOR_SIZE=0
        self._pb_path=pb

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():

            self._graph, self._sess = init(self._pb_path)
            self.img_input = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.outputs = tf.nn.softmax(tf.get_default_graph().get_tensor_by_name('tower_0/cls_output:0'))
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')



    def run(self,img):
        outputs = self._sess.run(
            self.outputs, feed_dict={self.img_input: img,self.training:False}
        )


        id=np.argmax(outputs)
        print(id)
        return  id







