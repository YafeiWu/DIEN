import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import numpy as np
import os
import sys
import time
from feature_def import UserSeqFeature
from random import randint

MinProbability = 0.00000001
EPR_THRESHOLD = 7.0
def base64_to_int32(base64string):
    decoded = tf.decode_base64(base64string)
    record = tf.decode_raw(decoded, tf.int32)
    return record

class BaseModel(object):
    def __init__(self, conf, task="train"):
        self.task = task
        self.n_uid = conf['n_uid']
        self.n_utype = conf['n_utype']
        self.n_mid = conf['n_mid']
        self.n_cat = conf['n_cat']
        self.n_tag = conf['n_tag']
        self.uid_embedding_dim = conf['uid_embedding_dim']
        self.utype_embedding_dim = conf['utype_embedding_dim']
        self.mid_embedding_dim = conf['mid_embedding_dim']
        self.cat_embedding_dim = conf['cat_embedding_dim']
        self.tag_embedding_dim = conf['tag_embedding_dim']
        self.enable_uid = conf['enable_uid']
        self.enable_tag = conf['enable_tag']
        self.hidden_size = conf['hidden_size']
        self.attention_size = conf['attention_size']
        self.training_data = conf['train_file']
        self.test_data = conf['test_file']
        self.batch_size = conf['batch_size']
        self.epochs = conf['epochs']
        self.maxLen = conf['maxlen']
        self.targetLen = conf['targetLen']
        self.negSeqLen = conf['negseq_length']
        self.enable_shuffle = conf['enable_shuffle']
        self.negStartIdx = 6+2*self.maxLen+1 #207
        self.use_negsampling = conf['use_negsampling']
        self.use_pair_loss = conf['use_pair_loss']
        self.fixTagsLen = conf['tags_length']
        self.feat_group = self.featGroup()
        self.item_feat_dim = self.mid_embedding_dim + self.cat_embedding_dim

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

    def fill_noclkseq(self, eb, eb_dim):
        eb_1 = tf.expand_dims(eb, 1)
        eb_2 = tf.tile(eb_1, multiples=[1, self.maxLen/tf.shape(eb)[1], 1, 1])
        eb_3 = tf.reshape(eb_2,
                          [-1, self.maxLen, eb_dim])
        return eb_3

    def reshape_multiseq(self, eb, eb_dim, multi_len, res_len):
        eb_1  = tf.reshape(eb,
                           [-1, res_len, multi_len, eb_dim])
        eb_2 = tf.reshape(eb_1,
                          [-1, res_len, multi_len * eb_dim ])
        return eb_2


    def featGroupV0(self):
        feat_names = [
            ("target",2), ## 0:2
            ("uid",1), ## 2
            ("utype",1), ## 3
            ("mid",1), ## 4
            ("cate",1), ## 5
            ("tags",self.fixTagsLen), ## 6:11
            ("clkseq_len",1), ## 11
            ("clkmid_seq",self.maxLen), ## 12:112
            ("clkcate_seq",self.maxLen), ## 112:212
            ("clktags_seq",self.maxLen*self.fixTagsLen), ## 212:712
            ("noclkseq_len",1), ## 712
            ("noclkmid_seq",self.negSeqLen), ## 713:718
            ("noclkcate_seq",self.negSeqLen), ## 718:723
            ("noclktags_seq",self.negSeqLen*self.fixTagsLen) ## 723:748
        ]
        feat_group = {}
        cur_offset = 0
        for feat_name, feat_width in feat_names:
            feat_group[feat_name] = UserSeqFeature(fname=feat_name,foffset=cur_offset,fends=cur_offset+feat_width,fwidth=feat_width)
            cur_offset += feat_width
            print("featGroupV0:\t{}".format(feat_group[feat_name]))
        return feat_group

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
        ]
        feat_group = {}
        cur_offset = 0
        for feat_name, feat_width in feat_names:
            feat_group[feat_name] = UserSeqFeature(fname=feat_name,foffset=cur_offset,fends=cur_offset+feat_width,fwidth=feat_width)
            cur_offset += feat_width
            print("featGroup:\t{}".format(feat_group[feat_name]))
        return feat_group

    def get_one_group(self, feats_batches, fname):
        foffset = getattr(self.feat_group[fname], 'f_offset')
        fends = getattr(self.feat_group[fname],'f_ends')
        fwidth = getattr(self.feat_group[fname],'f_width')
        # print("\tDEBUG fname :{}, fwidth: {}, foffset:{}, fends:{}".format(fname, fwidth, foffset, fends))
        if fwidth>1:
            return feats_batches[:, foffset: fends]
        else:
            return feats_batches[:, foffset]

    def prepare_from_base64(self, file, for_training=False):
        dataset = tf.data.TextLineDataset(file)
        if self.enable_shuffle and for_training:
            dataset = dataset.shuffle(buffer_size=self.batch_size * 500)
        if self.task == "train":
            dataset = dataset.repeat(self.epochs) if for_training else dataset.repeat()
        else:
            pass ### test task with no repeat
        dataset = dataset.map(lambda x: base64_to_int32(x), num_parallel_calls=64)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        batches = dataset.make_one_shot_iterator()
        return batches.get_next()


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
        self.y_hat = tf.nn.softmax(dnn3) + MinProbability

        with tf.name_scope('Metrics'):
            # Pair-wise loss and optimizer initialization
            if self.use_pair_loss:
                self.y_hat = self.y_hat[:,:,0] ### the 1st of 2d-softmax means positive probability
                pos_hat_ = tf.expand_dims(self.y_hat[:, 0], 1)
                neg_hat = self.y_hat[:, 1:tf.shape(self.y_hat)[1]]
                pos_hat = tf.tile(pos_hat_, multiples= [1, tf.shape(neg_hat)[1]])
                pair_prop = tf.sigmoid(pos_hat - neg_hat) + MinProbability
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

    def auxiliary_loss(self, h_states, click_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        r_slice = tf.convert_to_tensor(randint(0, self.batch_size))
        head_seq = tf.slice(click_seq, [r_slice, 0, 0], [tf.shape(click_seq)[0]-r_slice, tf.shape(click_seq)[1], tf.shape(click_seq)[2]])
        tail_seq = tf.slice(click_seq, [0, 0, 0], [r_slice, tf.shape(click_seq)[1], tf.shape(click_seq)[2]])
        noclick_seq =  tf.reshape(tf.concat([head_seq, tail_seq], 0),[self.batch_size, self.maxLen-1, self.item_feat_dim])
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        if self.use_pair_loss:
            ### pairwise loss
            pair_prop = tf.sigmoid(click_prop_ - noclick_prop_) + MinProbability ### 1 negative sample for each positive sample
            pair_loss = - tf.reshape(tf.log(pair_prop), [-1, tf.shape(click_seq)[1]]) * mask
            loss_ = tf.reduce_mean(pair_loss)
        else:
            ### ctr loss
            click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
            noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
            loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)

        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + MinProbability
        return y_hat

    def train(self, sess, inps):
        loss, aux_loss, top1_accuracy, target_accuracy, merged, _ = sess.run([self.loss, self.aux_loss, self.top1_accuracy, self.target_accuracy, self.merged, self.optimizer], feed_dict={
            self.for_training: inps[0],
            self.lr: inps[1]
        })
        return loss, aux_loss, top1_accuracy, target_accuracy, merged

    def test(self, sess, inps):
        loss, aux_loss, top1_accuracy, target_accuracy, merged = sess.run([self.loss, self.aux_loss, self.top1_accuracy, self.target_accuracy, self.merged], feed_dict={
            self.for_training: inps[0],
            self.lr: inps[1]
        })
        return loss, aux_loss, top1_accuracy, target_accuracy, merged

    def calculate(self, sess, inps):
        probs, targets, uids, loss, aux_loss, top1_accuracy, target_accuracy = sess.run([self.y_hat, self.target_ph, self.uid_batch_ph,
                                                                                         self.loss, self.aux_loss, self.top1_accuracy, self.target_accuracy], feed_dict={
            self.for_training: inps[0],
            self.lr: inps[1]
        })
        return probs, targets, uids, loss, aux_loss, top1_accuracy, target_accuracy

    def update_best_model(self, sess, path, iter):
        save_dir , prefix = os.path.split(path)
        files_ = os.listdir(save_dir)
        for file_ in files_:
            if file_.startswith(prefix):
                os.remove(os.path.join(save_dir,file_))
        self.save(sess, path+"--"+str(iter))

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


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
