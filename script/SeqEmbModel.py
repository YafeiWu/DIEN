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
            self.target_weight = self.target_ph / tf.constant(30.0, dtype=tf.float32) # use half minutes weight
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
                self.target_mask_pair = np.zeros((self.batch_size, self.targetLen-1), dtype=np.float32)
                ### enable pos_sample vs default
                self.target_mask_pair = tf.sequence_mask(self.target_len + tf.constant(1, tf.int32), self.target_mask_pair.shape[1], dtype=tf.float32)
                self.target_weight_pair = tf.tile(self.target_weight[:,0:1], [1, self.targetLen-1]) - self.target_weight[:,1:]
                self.target_weight_pair = self.target_mask_pair * self.target_weight_pair # fix shape for sequence_mask

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
        self.utype_batch_ph = tf.expand_dims(tf.cast(self.utype_batch_ph, dtype=tf.float32), -1)
        with tf.name_scope('Embedding_layer'):
            if self.enable_uid:
                self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [self.n_uid, self.uid_embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
                self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
                self.user_batch_embedded = tf.concat([self.uid_batch_embedded, self.utype_batch_ph], 1)
            else:
                self.user_batch_embedded = self.utype_batch_ph

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [self.n_mid, self.mid_embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [self.n_cat, self.cat_embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)

            if self.enable_tag:
                self.item_feat_dim += self.tag_embedding_dim * self.fixTagsLen
                self.tag_embeddings_var = tf.get_variable("tag_embedding_var", [self.n_tag, self.tag_embedding_dim], initializer=tf.contrib.layers.xavier_initializer())
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
            tf.summary.histogram('user_eb', self.user_eb)
            bn1 = tf.layers.batch_normalization(inputs=inp, name='user_bn1',training=self.for_training)
            tf.summary.histogram('user_bn1_output', bn1)
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='user_f1')
            tf.summary.histogram('user_f1_output', dnn1)
            dnn1 = prelu(dnn1, 'user_prelu1')
            return dnn1

    def build_item_vec(self, inp):
        with tf.name_scope('build_item_vec'):
            tf.summary.histogram('item_eb', self.item_eb)
            bn1 = tf.layers.batch_normalization(inputs=inp, name='item_bn1',training=self.for_training)
            tf.summary.histogram('item_bn1_output', bn1)
            dnn1 = tf.layers.dense(bn1, 100, activation=None, name='item_f1')
            tf.summary.histogram('item_f1_output', dnn1)
            dnn1 = prelu(dnn1, 'item_prelu1')
            return dnn1

    def user_cross_item(self):
        with tf.name_scope('user_cross_item'):
            self.user_vec = self.build_user_vec(self.user_eb)
            self.item_vec = self.build_item_vec(self.item_eb)
            self.user_vec_list = tf.tile(self.user_vec, [1, tf.shape(self.item_vec)[1]])
            self.user_vec_list = tf.reshape(self.user_vec_list, tf.shape(self.item_vec))
            self.user_vec_normal  = tf.sqrt(tf.reduce_sum(tf.square(self.user_vec_list), 2, keepdims=True))
            self.item_vec_normal = tf.sqrt(tf.reduce_sum(tf.square(self.item_vec), 2, keepdims=True))
            self.cross_raw = tf.reduce_sum(tf.multiply(self.user_vec_list, self.item_vec), 2, keepdims=True) / tf.multiply(self.user_vec_normal, self.item_vec_normal)


    def ctr_accuracy(self):
        # Generate label matrix
        target_ = tf.expand_dims(self.target_ph, -1) - tf.constant(EPR_THRESHOLD, dtype=tf.float32)
        label = tf.concat([target_, -target_], -1)
        ones_ = tf.ones_like(label)
        zeros_ = tf.zeros_like(label)
        label = tf.where(label>0.0, x=ones_, y=zeros_)

        # Accuracy metric
        accuracy_masked = tf.cast(tf.equal(tf.round(self.y_hat[:,:,0]), label[:,:,0]), tf.float32) * self.target_mask
        accuracy_ = tf.reduce_sum(accuracy_masked) / tf.reduce_sum(self.target_mask)
        tf.summary.scalar('ctr_accuracy', accuracy_)

        return label, accuracy_

    def metrics(self):
        with tf.name_scope('Metrics'):
            self.y_hat = tf.nn.softmax(self.cross_raw) + epsilon
            self.label, self.ctr_accuracy = self.ctr_accuracy()
            # Pair-wise loss and optimizer initialization
            if self.use_pair_loss:
                self.cross_sim = tf.reshape(self.cross_raw, [-1, tf.shape(self.cross_raw)[1]])
                tf.summary.histogram('cross_sim', self.cross_raw)
                neg_hat = self.cross_sim[:, 1:tf.shape(self.cross_sim)[1]]
                pos_hat = tf.tile(self.cross_sim[:, 0:1], [1, tf.shape(neg_hat)[1]])
                pair_prop = tf.sigmoid(pos_hat - neg_hat)
                pair_loss_ = -tf.reshape(tf.log(pair_prop+ epsilon), [-1, tf.shape(neg_hat)[1]]) * self.target_mask_pair

                pair_loss = tf.reduce_sum(pair_loss_ * self.target_weight_pair) / tf.reduce_sum(self.target_mask_pair)
                tf.summary.scalar('pair_loss', pair_loss)
                self.loss = pair_loss

                # Accuracy metric* self.target_mask
                target_ = tf.ones(shape=(tf.shape(neg_hat)[0], tf.shape(neg_hat)[1]), dtype=np.float32)

                pair_acc = tf.reduce_sum(tf.cast(tf.equal(tf.round(pair_prop-epsilon), target_), tf.float32) * self.target_mask_pair)
                self.target_accuracy = pair_acc / tf.reduce_sum(self.target_mask_pair)
                tf.summary.scalar('pair_accuracy', self.target_accuracy)

                top1_prediction = tf.equal(tf.argmax(self.cross_sim * self.target_mask, 1), 0)
                self.top1_accuracy = tf.reduce_mean(tf.cast(top1_prediction, tf.float32))
                tf.summary.scalar('top1_accuracy', self.top1_accuracy)

                top3_prediction = tf.less(tf.argmax(self.cross_sim * self.target_mask, 1), 3)
                self.top3_accuracy = tf.reduce_mean(tf.cast(top3_prediction, tf.float32))
                tf.summary.scalar('top3_accuracy', self.top3_accuracy)

            else:
                # Cross-entropy loss and optimizer initialization
                ctr_loss_w = tf.reduce_sum(tf.log(self.y_hat) * self.label, 2) * self.target_mask * self.target_weight
                ctr_loss = - tf.reduce_sum(ctr_loss_w) / tf.reduce_sum(self.target_mask)
                tf.summary.scalar('ctr_loss', ctr_loss)
                self.loss = ctr_loss

                correct_prediction = tf.equal(tf.argmax(self.y_hat[:,:,0] * self.target_mask, 1), 0)
                self.top1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('top1_accuracy', self.top1_accuracy)

            if self.use_negsampling:
                self.loss += self.aux_loss
                tf.summary.scalar('aux_loss', self.aux_loss)

            tf.summary.scalar('loss', self.loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                raw_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.optimizer = raw_optimizer.minimize(self.loss)
                self.grads = raw_optimizer.compute_gradients(self.loss)
                for g in self.grads:
                    tf.summary.histogram("%s-grad" % g[1].name, g[0])

                self.optimizer = raw_optimizer.apply_gradients(self.grads)

        self.merged = tf.summary.merge_all()
