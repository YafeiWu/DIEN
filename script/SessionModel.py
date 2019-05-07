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

class SessionModel(BaseModel):
    """SessioinModel"""
    def __init__(self, conf, task="train"):
        super(SessionModel, self).__init__(conf,task)
        self.for_training = tf.placeholder_with_default(tf.constant(False),shape=(),name="training_flag")
        self.lr = tf.placeholder(tf.float64, [],name="learning_rate")
        self.use_negsampling = False

        self.global_embedding_size = 32
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.eb_var = {}

        self.feat_group = []
        self.label = None
        self.target_ph = None
        self.weight = None
        self.user_feats = None
        self.item_feats = None
        self.item_his_eb = None
        self.item_his_eb_sum = None
        self.fc_inputs = None

        self.y_hat = None
        self.optimizer = None
        self.loss = None
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.l1_loss = tf.constant(0., dtype=tf.float32)
        self.l2_loss = tf.constant(0., dtype=tf.float32)
        self.merged = None
        self.top1_accuracy = tf.constant(0., dtype=tf.float32)
        self.target_accuracy = tf.constant(0., dtype=tf.float32)


    def run(self):
        self.featGroup()
        self.pnn_processor()
        self.build_fcn_net()
        self.metrics()

    def reset_features(self):
        self.label = None
        self.weight = None
        self.user_feats = None
        self.item_feats = None
        self.item_his_eb = None
        self.item_his_eb_sum = None
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.l1_loss = tf.constant(0., dtype=tf.float32)
        self.l2_loss = tf.constant(0., dtype=tf.float32)

    def featGroup(self):
        self.feat_group = [
            UserSeqFeature(fname="label", f_group='label',f_offset=0, f_ends=0, f_width=1, f_type=tf.float32, f_seqsize=1, fembedding=False),
            UserSeqFeature(fname="user_id", f_group='uid',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_uid),
            UserSeqFeature(fname="tar_weight", f_group='stats', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.float32, f_seqsize=1, fembedding=False),
            UserSeqFeature(fname="tar_item", f_group='vid', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_mid),
            UserSeqFeature(fname="tar_cat", f_group='category',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_cat),
            UserSeqFeature(fname="tar_tags", f_group='tag',f_offset=-1, f_ends=-1, f_width=self.fixTagsLen, f_type=tf.int32, f_seqsize=1, f_max=self.n_tag),
            UserSeqFeature(fname="seq_item", f_group='vid',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_mid),
            UserSeqFeature(fname="seq_cat", f_group='category',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_cat),
            UserSeqFeature(fname="seq_tags", f_group='tag',f_offset=-1, f_ends=-1, f_width=self.fixTagsLen, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_tag)
        ]
        cur_offset = 0
        for u_feat in self.feat_group:
            u_feat.f_offset = cur_offset
            u_feat.f_ends = cur_offset + u_feat.f_width
            cur_offset += u_feat.f_width
            print("INFO -> Raw feature :\t{}".format(u_feat))

    def get_one_group(self, feats_batches, u_feat):
        f_offset = getattr(u_feat, 'f_offset')
        f_ends = getattr(u_feat,'f_ends')
        f_width = getattr(u_feat,'f_width')
        f_type = getattr(u_feat,'f_type')
        print("\tDEBUG f_name :{}, f_width: {}, f_offset:{}, f_ends:{}".format(f_name, f_width, f_offset, f_ends))
        feat = feats_batches[:, f_offset: f_ends] if f_width>1 else feats_batches[:, f_offset]
        if f_type != tf.int32:
            return tf.cast(feat, dtype=f_type)
        else:
            feat

    def pnn_processor(self):
        with tf.name_scope('Input'):
            train_batches = self.prepare_from_base64(self.training_data, for_training=True)
            test_batches = self.prepare_from_base64(self.test_data, for_training=False)
            feats_batches = tf.cond(self.for_training, lambda:train_batches, lambda:test_batches)

        with tf.name_scope("Embedding"):
            self.reset_features()

            for f in self.feat_group:
                batch_ph = self.get_one_group(feats_batches, f.f_name)
                if f.f_name == "label":
                    self.label = batch_ph
                elif f.f_name == "weight":
                    self.weight = batch_ph

                if not f.f_embedding:
                    continue

                eb_name = f.f_group + "_embedding"
                if f.f_group not in self.eb_var:
                    self.eb_var[eb_name] =  tf.get_variable("{}_var".format(eb_name), [self.f_max, self.global_embedding_size], initializer=tf.contrib.layers.xavier_initializer())
                raw_embedding = tf.nn.embedding_lookup(self.eb_var[eb_name], batch_ph)
                if f.f_seqsize >1:
                    ### aggregate user's item history features
                    if f.f_width >1:
                        raw_embedding_ = tf.reshape(raw_embedding, [-1, f.f_seqsize, f.f_width, self.global_embedding_size])
                        embedding = tf.reduce_mean(raw_embedding_, 2)
                    else:
                        embedding = raw_embedding

                    if self.item_his_eb is None:
                        self.item_his_eb = embedding
                    else:
                        self.item_his_eb = tf.concat([self.item_his_eb, embedding], 2)

                else:
                    ### aggregate user/item features
                    if f.f_width >1:
                        embedding = tf.reduce_mean(raw_embedding, 1)
                    else:
                        embedding = raw_embedding

                    if f.f_name.startswith("user"):
                        # aggregate user features
                        if self.user_feats is None:
                            self.user_feats = embedding
                        else:
                            self.user_feats = tf.concat([self.user_feats, embedding], 1)
                    else:
                        # aggregate item features
                        if self.item_feats is None:
                            self.item_feats = embedding
                        else:
                            self.item_feats = tf.concat([self.item_feats, embedding], 1)

        with tf.name_scope("Concat"):
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
            self.fc_inputs = tf.concat([self.user_feats, self.item_feats, self.item_his_eb_sum,
                                  self.item_feats * self.item_his_eb_sum], 1)

    def expand_label(self):
        target_ = tf.expand_dims(self.weight, -1) - tf.constant(EPR_THRESHOLD, dtype=tf.float32)
        label = tf.concat([target_, -target_], -1)
        ones_ = tf.ones_like(label)
        zeros_ = tf.zeros_like(label)
        target_ph_ = tf.where(label>0.0, x=ones_, y=zeros_)
        return target_ph_

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

    def metrics(self):
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.expand_label()
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss

            # L1/L2 loss
            for embedding_var in self.eb_var:
                self.l1_loss += tf.reduce_mean(tf.abs(embedding_var))
                self.l2_loss += tf.reduce_mean(tf.nn.l2_loss(embedding_var))
            self.loss += self.l1_loss
            self.loss += self.l2_loss

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('l1_loss', self.l1_loss)
            tf.summary.scalar('l2_loss', self.l2_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.target_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('target_accuracy', self.target_accuracy)

        self.merged = tf.summary.merge_all()
