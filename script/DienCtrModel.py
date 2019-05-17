import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import GRUCell
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from BaseModel import BaseModel
from feature_def import UserSeqFeature
from datasource import get_one_group,expand_label
from datasource import epsilon

class DienCtrModel(BaseModel):
    """DienCtrModel"""
    def __init__(self, conf="config_dien_ctr.yml", task="train"):
        super(DienCtrModel, self).__init__(conf, task)
        self.for_training = tf.placeholder_with_default(tf.constant(False),shape=(),name="training_flag")
        self.lr = tf.placeholder(tf.float64, [],name="learning_rate")

        ### model config
        self.task = task
        self.n_uid = conf['n_uid']
        self.n_mid = conf['n_mid']
        self.n_cat = conf['n_cat']
        self.n_tag = conf['n_tag']
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
        self.fixTagsLen = conf['tags_length']


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

        self.y_hat = None
        self.optimizer = None
        self.loss = None
        self.aux_loss = tf.constant(0., dtype=tf.float32)
        self.l1_losses = []
        self.l2_losses = []
        self.top1_accuracy = tf.constant(0., dtype=tf.float32)
        self.target_accuracy = tf.constant(0., dtype=tf.float32)
        self.merged = None
        self.mask = None

        self.run()

    def feat_group(self):
        self.feat_group = [
            UserSeqFeature(f_name="label", f_group='label',f_offset=0, f_ends=0, f_width=1, f_type=tf.float32, f_seqsize=1, f_embedding=False),
            UserSeqFeature(f_name="user_id", f_group='uid',f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_uid),
            UserSeqFeature(f_name="tar_weight", f_group='stats', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.float32, f_seqsize=1, f_embedding=False),
            UserSeqFeature(f_name="tar_item", f_group='vid', f_offset=-1, f_ends=-1, f_width=1, f_type=tf.int32, f_seqsize=1, f_max=self.n_mid),
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

    def run(self):
        self.feat_group()
        self.parse2embedding()
        self.build_model()
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

    def neg_sample(self, batch_ph):
        candidates = tf.reshape(batch_ph, [1, -1])
        idx = tf.where(candidates > 0)
        clicks = tf.gather_nd(candidates, idx)
        shuffled_clicks = tf.random.shuffle(clicks)


    def parse2embedding(self):
        with tf.name_scope('Input'):
            train_batches = self.prepare_from_base64(self.training_data, for_training=True)
            test_batches = self.prepare_from_base64(self.test_data, for_training=False)
            feats_batches = tf.cond(self.for_training, lambda:train_batches, lambda:test_batches)

        with tf.name_scope("Embedding"):
            self.reset_features()

            for f in self.feat_group:
                if f.f_mask:
                    continue
                batch_ph = get_one_group(feats_batches, f)
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
                    self.mask = tf.sequence_mask(batch_ph, mask.shape[1], dtype=tf.float32)

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


    def build_model(self):
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
        #                                  self.noclk_item_his_eb[:, 1:, :],
        #                                  self.mask[:, 1:], stag="gru")
        # self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, self.attention_size, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.hidden_size), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.user_feats, self.item_feats, self.item_his_eb_sum, self.item_feats * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
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
        y_hat = tf.nn.softmax(dnn3) + epsilon
        return y_hat

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
        self.y_hat = tf.nn.softmax(dnn3) + epsilon

    def metrics(self):
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = expand_label(self.weight)
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss


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
