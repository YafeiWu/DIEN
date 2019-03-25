import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from BaseModel import BaseModel
import numpy as np
import os
import sys
import time
from feature_def import UserSeqFeature
from random import randint

class SeqEmbModel(BaseModel):
    """DIEN"""
    def __init__(self, conf, task="train"):
        super(SeqEmbModel, self).__init__(conf,task)

        with tf.name_scope('concat_user_seq'):
            his_weights = tf.expand_dims(self.u_his_weight, -1)
            his_weights = tf.tile(his_weights, [1,1, tf.shape(self.item_his_eb)[2]])
            his_seq_sum = self.item_his_eb * his_weights
            u_his_inp = tf.concat(self.user_batch_embedded, his_seq_sum)
            item_his_eb_sum_  = tf.expand_dims(self.item_his_eb_sum, 1)
            item_his_eb_sum_ = tf.tile(item_his_eb_sum_, multiples=[1, tf.shape(self.item_eb)[1], 1])
            u_now_inp = tf.concat([self.item_eb, self.item_eb * item_his_eb_sum_], 2)
            u_his_inp_exp = tf.expand_dims(u_his_inp, 1)
            u_his_inp_exp = tf.tile(u_his_inp_exp, multiples=[1, tf.shape(u_now_inp)[1], 1])
            fcn_inp = tf.concat([u_now_inp, u_his_inp_exp], 2)

            self.use_negsampling = False
            self.build_fcn_net(fcn_inp, use_dice=True)


        # inp = tf.concat([self.user_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        # self.build_fcn_net(inp, use_dice=True)
