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

class DIENModel(BaseModel):
    """DIEN"""
    def __init__(self, conf, task="train"):
        super(DIENModel, self).__init__(conf,task)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        if self.use_negsampling:

            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1
        else:
            self.aux_loss = tf.convert_to_tensor(0.)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, self.attention_size, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            idx_ = tf.constant(0)
            rnn_outputs2_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            final_state2_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            def target_loop_cond(idx_, rnn_outputs2_list, final_state2_list):
                return idx_ < self.targetLen

            def target_loop_body(idx_, rnn_outputs2_list, final_state2_list):
                rnn_outputs2_, final_state2_ = \
                    dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                att_scores = tf.expand_dims(alphas[:,idx_,:], -1),
                                sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                scope="gru2")
                rnn_outputs2_list = rnn_outputs2_list.write(idx_, rnn_outputs2_)
                final_state2_list = final_state2_list.write(idx_, final_state2_)

                return idx_+1, rnn_outputs2_list, final_state2_list


            _, rnn_outputs2, final_state2 = tf.while_loop(target_loop_cond, target_loop_body,
                                                                    [idx_ , rnn_outputs2_list, final_state2_list])
            rnn_outputs2 = tf.transpose(rnn_outputs2.stack(), [1,0,2,3])
            final_state2 = tf.transpose(final_state2.stack(), [1,0,2])

            tf.summary.histogram('GRU2_rnn_outputs2', rnn_outputs2)
            tf.summary.histogram('GRU2_Final_State', final_state2)

        with tf.name_scope('expand4listwise'):
            u_his_inp = tf.concat([self.user_batch_embedded, self.item_his_eb_sum], 1)
            item_his_eb_sum_  = tf.expand_dims(self.item_his_eb_sum, 1)
            item_his_eb_sum_ = tf.tile(item_his_eb_sum_, multiples=[1, tf.shape(self.item_eb)[1], 1])
            u_now_inp = tf.concat([self.item_eb, self.item_eb * item_his_eb_sum_, final_state2], 2)
            u_his_inp_exp = tf.expand_dims(u_his_inp, 1)
            u_his_inp_exp = tf.tile(u_his_inp_exp, multiples=[1, tf.shape(u_now_inp)[1], 1])
            fcn_inp = tf.concat([u_now_inp, u_his_inp_exp], 2)

            self.build_fcn_net(fcn_inp, use_dice=True)

        # inp = tf.concat([self.user_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        # self.build_fcn_net(inp, use_dice=True)
