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
    """SEMB"""
    def __init__(self, conf, task="train"):
        super(SeqEmbModel, self).__init__(conf,task)
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.feat_group = self.featGroup()
        self.inputsLayer()
        self.embeddingLayer()
        self.concatLayer()

    def featGroup(self):
        feat_names = [
            ("uid", 1), ## 0
            ("utype", 1), ## 1:11
            ("target_len", 1), ## 1:11
            ("target_weight", self.targetLen), ## 1:11
            ("target_mids", self.targetLen), ## 3
            ("target_cats", self.targetLen), ## 4
            ("target_tags", self.targetLen * self.fixTagsLen), ## 5
            ("clkseq_len", 1), ## 11
            ("clkmid_seq", self.maxLen), ## 12:112
            ("clkcate_seq", self.maxLen), ## 112:212
            ("clktags_seq", self.maxLen * self.fixTagsLen), ## 212:712
            ("clkweight_seq", self.maxLen), ## 212:712
        ]
        feat_group = {}
        cur_offset = 0
        for feat_name, feat_width in feat_names:
            feat_group[feat_name] = UserSeqFeature(fname=feat_name,foffset=cur_offset,fends=cur_offset+feat_width,fwidth=feat_width)
            cur_offset += feat_width
            print("featGroup:\t{}".format(feat_group[feat_name]))
        return feat_group

    def inputsLayer(self):
        with tf.name_scope('Inputs'):
            self.for_training = tf.placeholder_with_default(tf.constant(False),shape=(),name="training_flag")
            self.lr = tf.placeholder(tf.float64, [],name="learning_rate")
            train_batches = self.prepare_from_base64(self.training_data, for_training=True)
            test_batches = self.prepare_from_base64(self.test_data, for_training=False)
            feats_batches = tf.cond(self.for_training, lambda:train_batches, lambda:test_batches)

            self.target_ph = tf.cast(self.get_one_group(feats_batches, 'target_weight'), dtype=tf.float32)
            self.target_weight = self.target_ph / tf.constant(60.0, dtype=tf.float32) # use minutes weight
            self.target_weight = tf.clip_by_value(self.target_weight,1,10)
            self.uid_batch_ph = self.get_one_group(feats_batches, 'uid')
            self.utype_batch_ph = self.get_one_group(feats_batches, 'utype')
            self.target_len = self.get_one_group(feats_batches, 'target_len')
            self.mid_batch_ph = self.get_one_group(feats_batches, 'target_mids')
            self.cat_batch_ph = self.get_one_group(feats_batches, 'target_cats')
            self.seq_len_ph = self.get_one_group(feats_batches, 'clkseq_len')
            self.mid_his_batch_ph = self.get_one_group(feats_batches, 'clkmid_seq')
            self.cat_his_batch_ph = self.get_one_group(feats_batches, 'clkcate_seq')
            self.weight_his_batch_ph = tf.cast(self.get_one_group(feats_batches, 'clkweight_seq'), dtype=tf.float32)

            if self.enable_tag:
                self.tags_batch_ph = self.get_one_group(feats_batches, 'target_tags')
                self.tags_his_batch_ph = self.get_one_group(feats_batches, 'clktags_seq')

            if self.use_pair_loss:
                self.target_mask = np.zeros((self.batch_size, self.targetLen-1), dtype=np.float32)
                self.target_mask = tf.sequence_mask(self.target_len, self.target_mask.shape[1], dtype=tf.float32)
                self.target_weight = tf.tile(self.target_weight[:,0:1], [1, self.targetLen-1]) - self.target_weight[:,1:]
                self.target_weight = self.target_mask * self.target_weight # fix shape for sequence_mask
            else:
                self.target_mask = np.zeros((self.batch_size, self.targetLen), dtype=np.float32)
                self.target_mask = tf.sequence_mask(self.target_len, self.target_mask.shape[1], dtype=tf.float32)

            self.mask = np.zeros((self.batch_size, self.maxLen), dtype=np.float32)
            self.mask = tf.sequence_mask(self.seq_len_ph, self.mask.shape[1], dtype=tf.float32)

    def embeddingLayer(self):
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.utype_embeddings_var = tf.get_variable("utype_embedding_var", [self.n_utype, self.utype_embedding_dim], initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram('utype_embedding_var', self.utype_embeddings_var)
            self.utype_batch_embedded = tf.nn.embedding_lookup(self.utype_embeddings_var, self.utype_batch_ph)
            if self.enable_uid:
                self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [self.n_uid, self.uid_embedding_dim], initializer=tf.random_normal_initializer(stddev=0.01))
                tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
                self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
                self.user_batch_embedded = tf.concat([self.uid_batch_embedded, self.utype_batch_embedded], 1)
            else:
                self.user_batch_embedded = self.utype_batch_embedded

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [self.n_mid, self.mid_embedding_dim], initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [self.n_cat, self.cat_embedding_dim], initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)

            if self.enable_tag:
                self.item_feat_dim += self.tag_embedding_dim * self.fixTagsLen
                self.tag_embeddings_var = tf.get_variable("tag_embedding_var", [self.n_tag, self.tag_embedding_dim], initializer=tf.random_normal_initializer(stddev=0.01))
                tf.summary.histogram('tag_embeddings_var', self.tag_embeddings_var)

                self.tags_batch_embedded = tf.nn.embedding_lookup(self.tag_embeddings_var, self.tags_batch_ph)
                self.tags_batch_embedded = self.reshape_multiseq(self.tags_batch_embedded, self.tag_embedding_dim, self.fixTagsLen, self.targetLen)

                self.tags_his_batch_embedded = tf.nn.embedding_lookup(self.tag_embeddings_var, self.tags_his_batch_ph)
                self.tags_his_batch_embedded = self.reshape_multiseq(self.tags_his_batch_embedded, self.tag_embedding_dim, self.fixTagsLen, self.maxLen)

                self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded, self.tags_batch_embedded], 2)
                self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded, self.tags_his_batch_embedded], 2)
            else:
                self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 2)
                self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)

            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

    def concatLayer(self):
        with tf.name_scope('concat_user_seq'):
            his_weights = tf.expand_dims(self.weight_his_batch_ph, -1)
            his_weights = tf.tile(his_weights, [1, 1, tf.shape(self.item_his_eb)[2]])
            his_seq_sum = tf.reduce_sum(self.item_his_eb * his_weights, 1)
            u_his_inp = tf.concat([self.user_batch_embedded, his_seq_sum], 1)
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
