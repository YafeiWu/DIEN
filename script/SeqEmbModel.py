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

epsilon = 0.000000001
EPR_THRESHOLD = 7.0

class SeqEmbModel(BaseModel):
    """SEMB"""
    def __init__(self, conf, task="train"):
        super(SeqEmbModel, self).__init__(conf,task)
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.feat_group = self.featGroup()
        self.inputsLayer()
        self.build_model()

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

    def build_model(self):
        self.embeddingLayer()
        self.concatLayer()
        self.use_negsampling = False
        self.user_cross_item()
        self.metrics()

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

            self.mask = tf.expand_dims(self.mask, -1)
            self.mask = tf.tile(self.mask, [1, 1, tf.shape(self.item_his_eb)[2]])
            self.item_his_eb = self.item_his_eb * self.mask

            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

    def concatLayer(self):
        with tf.name_scope('concat_user_seq'):
            his_weights = tf.expand_dims(self.weight_his_batch_ph, -1)
            his_weights = tf.tile(his_weights, [1, 1, tf.shape(self.item_his_eb)[2]])
            his_seq_sum = tf.reduce_sum(self.item_his_eb * his_weights, 1)
            self.user_eb = tf.concat([self.user_batch_embedded, his_seq_sum, self.item_his_eb_sum], 1)

    def build_user_vec(self, inp):
        with tf.name_scope('build_user_vec'):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='user_bn1')
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='user_f1')
            dnn1 = prelu(dnn1, 'user_prelu1')
            return dnn1

    def build_item_vec(self, inp):
        with tf.name_scope('build_item_vec'):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='item_bn1')
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='item_f1')
            dnn1 = prelu(dnn1, 'item_prelu1')
            return dnn1

    def user_cross_item(self):
        with tf.name_scope('user_cross_item'):
            self.user_vec = self.build_user_vec(self.user_eb)
            self.item_vec = self.build_item_vec(self.item_eb)
            self.user_vec_list = tf.tile(self.user_vec, [1, tf.shape(self.item_vec)[1]])
            self.user_vec_list = tf.reshape(self.user_vec_list, tf.shape(self.item_vec))
            cross_raw = tf.multiply(self.user_vec_list, self.item_vec)
            bn1 = tf.layers.batch_normalization(inputs=cross_raw, name='bn1')
            dnn1 = tf.layers.dense(bn1, 50, activation=None, name='f1')
            dnn1 = prelu(dnn1, 'prelu1')
            dnn2 = tf.layers.dense(dnn1, 20, activation=None, name='f2' )
            dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
            self.y_hat = tf.nn.softmax(dnn3) + epsilon

    def metrics(self):
        with tf.name_scope('Metrics'):
            # Pair-wise loss and optimizer initialization
            if self.use_pair_loss:
                self.y_hat = self.y_hat[:,:,0] ### the 1st of 2d-softmax means positive probability
                pos_hat_ = tf.expand_dims(self.y_hat[:, 0], 1)
                neg_hat = self.y_hat[:, 1:tf.shape(self.y_hat)[1]]
                pos_hat = tf.tile(pos_hat_, multiples= [1, tf.shape(neg_hat)[1]])
                pair_prop = tf.sigmoid(pos_hat - neg_hat) + epsilon
                pair_loss_ = -tf.reshape(tf.log(pair_prop), [-1, tf.shape(neg_hat)[1]]) * self.target_mask

                pair_loss = tf.reduce_mean(pair_loss_ * self.target_weight)
                tf.summary.scalar('pair_loss', pair_loss)
                self.loss = pair_loss

                # Accuracy metric* self.target_mask
                target_ = tf.ones(shape=(tf.shape(neg_hat)[0], tf.shape(neg_hat)[1]), dtype=np.float32) * self.target_mask
                self.target_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(pair_prop), target_), tf.float32))
                tf.summary.scalar('pair_accuracy', self.target_accuracy)

            else:
                # Cross-entropy loss and optimizer initialization
                target_ = tf.expand_dims(self.target_ph, -1) - tf.constant(EPR_THRESHOLD, dtype=tf.float32)
                self.label = tf.concat([target_, -target_], -1)
                ones_ = tf.ones_like(self.label)
                zeros_ = tf.zeros_like(self.label)
                self.label = tf.where(self.label>0.0, x=ones_, y=zeros_)
                ctr_loss_w = tf.reduce_sum(tf.log(self.y_hat) * self.label, 2) * self.target_mask
                ctr_loss = - tf.reduce_mean(ctr_loss_w)
                tf.summary.scalar('ctr_loss', ctr_loss)
                self.loss = ctr_loss

                # Accuracy metric
                self.target_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat[:,:,0]), self.label[:,:,0]), tf.float32) * self.target_mask )
                tf.summary.scalar('ctr_accuracy', self.target_accuracy)

            if self.use_negsampling:
                self.loss += self.aux_loss
                tf.summary.scalar('aux_loss', self.aux_loss)

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            correct_prediction = tf.equal(tf.argmax(self.y_hat[:,:,0], 1), 0)
            self.top1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('top1_accuracy', self.top1_accuracy)

        self.merged = tf.summary.merge_all()
