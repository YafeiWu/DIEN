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
        self.uid_batch_ph = None
        self.user_feats = None
        self.item_feats = None
        self.item_his_eb = None
        self.item_his_eb_sum = None
        self.fc_inputs = None

        self.y_hat = None
        self.optimizer = None
        self.loss = None
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.l1_losses = []
        self.l2_losses = []
        self.top1_accuracy = tf.constant(0., dtype=tf.float32)
        self.target_accuracy = tf.constant(0., dtype=tf.float32)
        self.merged = None

        self.run()


    def run(self):
        self.featGroup()
        self.pnn_processor()
        self.build_fcn_one(self.fc_inputs)
        self.metrics()

    def reset_features(self):
        self.label = None
        self.weight = None
        self.user_feats = None
        self.item_feats = None
        self.item_his_eb = None
        self.item_his_eb_sum = None
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.l1_losses = []
        self.l2_losses = []

    def featGroup(self):
        self.feat_group = [
            UserSeqFeature(f_name="label", f_group='label',f_offset=0, f_ends=0, f_width=1, f_type=tf.float32, f_seqsize=1, f_embedding=False),
            UserSeqFeature(f_name="user_id", f_group='uid',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_uid),
            UserSeqFeature(f_name="tar_weight", f_group='stats', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.float32, f_seqsize=1, f_embedding=False),
            UserSeqFeature(f_name="tar_item", f_group='vid', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_mid)
            UserSeqFeature(f_name="tar_cat", f_group='category',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_cat, f_mask=True),
            UserSeqFeature(f_name="tar_tags", f_group='tag',f_offset=-1, f_ends=-1, f_width=self.fixTagsLen, f_type=tf.int32, f_seqsize=1, f_max=self.n_tag, f_mask=True),
            UserSeqFeature(f_name="seq_size", f_group='stats',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_embedding=False),
            UserSeqFeature(f_name="seq_item", f_group='vid',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_mid),
            UserSeqFeature(f_name="seq_cat", f_group='category',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_cat, f_mask=True),
            UserSeqFeature(f_name="seq_tags", f_group='tag',f_offset=-1, f_ends=-1, f_width=self.fixTagsLen, f_type=tf.int32, f_seqsize=self.maxLen, f_max=self.n_tag, f_mask=True),
            UserSeqFeature(f_name="seq_weights", f_group='stats',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=self.maxLen, f_embedding=False)
        ]
        cur_offset = 0
        for u_feat in self.feat_group:
            u_feat.f_offset = cur_offset
            u_feat.f_ends = cur_offset + u_feat.f_width * u_feat.f_seqsize
            cur_offset = u_feat.f_ends
            print("INFO -> Raw Feature Group :\t{}".format(u_feat))

    def get_one_group(self, feats_batches, f):
        print("DEBUG INFO -> f_name :{}, f_width: {}, f_seqsize:{}, f_offset:{}, f_ends:{}".format(f.f_name, f.f_width, f.f_seqsize, f.f_offset, f.f_ends))
        feat = feats_batches[:, f.f_offset: f.f_ends] if f.f_width * f.f_seqsize >1 else feats_batches[:, f.f_offset]
        if f.f_type != tf.int32:
            return tf.cast(feat, dtype=f.f_type)
        else:
            return feat

    def pnn_processor(self):
        with tf.name_scope('Input'):
            train_batches = self.prepare_from_base64(self.training_data, for_training=True)
            test_batches = self.prepare_from_base64(self.test_data, for_training=False)
            feats_batches = tf.cond(self.for_training, lambda:train_batches, lambda:test_batches)

        with tf.name_scope("Embedding"):
            self.reset_features()

            for f in self.feat_group:
                if f.f_mask:
                    continue
                batch_ph = self.get_one_group(feats_batches, f)
                if f.f_name == "label":
                    self.label = batch_ph
                    label_sum = tf.reduce_sum(batch_ph)/self.batch_size
                    print_tensor = tf.Print(label_sum, [label_sum], message="DEBUG INFO -> label_sum : ")
                    print("DEBUG INFO -> label_sum  : {}".format(print_tensor.eval()))

                elif f.f_name == "user_id":
                    self.uid_batch_ph = batch_ph
                elif f.f_name == "tar_weight":
                    self.weight = batch_ph
                elif f.f_name == "seq_size":
                    mask = np.zeros((self.batch_size, self.maxLen), dtype=np.float32)
                    mask = tf.sequence_mask(batch_ph, mask.shape[1], dtype=tf.float32)

                print_tensor = tf.Print(batch_ph, [batch_ph], message="DEBUG INFO -> This is one {} batch_ph : ".format(f.f_name))
                try:
                    batch_shape = print_tensor.eval().shape
                except Exception as e:
                    batch_shape = print_tensor.eval().length
                print("DEBUG INFO -> This is one {} batch_ph shape : {} \n{}".format(f.f_name, batch_shape, print_tensor.eval()))

                if not f.f_embedding:
                    continue

                eb_name = f.f_group + "_em_var"
                if eb_name not in self.eb_var:
                    self.eb_var[eb_name] =  tf.get_variable(eb_name, [f.f_max, self.global_embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.01), regularizer=self.regularizer)

                raw_embedding = tf.nn.embedding_lookup(self.eb_var[eb_name], batch_ph)
                # self.l1_losses.append(tf.reduce_mean(tf.abs(raw_embedding)))
                # self.l2_losses.append(tf.nn.l2_loss(raw_embedding)/self.batch_size)
                tf.summary.histogram('{}_em'.format(f.f_name), raw_embedding)
                tf.summary.histogram(eb_name, self.eb_var[eb_name])
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
            ### mask non_clciked history
            mask = tf.expand_dims(mask, -1)
            mask = tf.tile(mask, [1, 1, tf.shape(self.item_his_eb)[2]])
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb * mask, 1)
            self.fc_inputs = tf.concat([self.user_feats, self.item_feats, self.item_his_eb_sum,
                                  self.item_feats * self.item_his_eb_sum], 1)

    def expand_label(self):
        target_ = tf.expand_dims(self.weight, -1) - tf.constant(EPR_THRESHOLD, dtype=tf.float32)
        label = tf.concat([target_, -target_], -1)
        ones_ = tf.ones_like(label)
        zeros_ = tf.zeros_like(label)
        target_ph_ = tf.where(label>0.0, x=ones_, y=zeros_)
        return target_ph_

    def build_fcn_one(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 2, activation=None, name='f1', kernel_regularizer=self.regularizer)
        self.y_hat = tf.nn.softmax(dnn1) + 0.00000001

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


            # l1_loss = tf.reduce_mean(self.l1_losses)
            # l2_loss = tf.reduce_mean(self.l2_losses)
            reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.add_n(reg_set)
            self.loss += l2_loss

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('ctr_loss', ctr_loss)
            tf.summary.scalar('l2_loss', l2_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.target_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('target_accuracy', self.target_accuracy)

        self.merged = tf.summary.merge_all()
