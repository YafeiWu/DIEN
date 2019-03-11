import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
import numpy


def base64_to_int(base64string):
    decoded = tf.decode_base64(base64string)
    record = tf.decode_raw(decoded, tf.int32)
    return record

class BaseModel(object):
    def __init__(self, conf):
        self.n_uid = conf['n_uid']
        self.n_mid = conf['n_mid']
        self.n_cat = conf['n_cat']
        self.uid_embedding_dim = conf['uid_embedding_dim']
        self.mid_embedding_dim = conf['mid_embedding_dim']
        self.cat_embedding_dim = conf['cat_embedding_dim']
        self.training_data = conf['training_files']
        self.test_data = conf['test_files']
        self.batch_size = conf['batch_size']
        self.epochs = conf['epochs']
        self.maxLen = conf['max_length']

        with tf.name_scope('Inputs'):
            self.for_training = tf.placeholder_with_default(tf.constant(False),shape=(),name="training_flag")
            self.lr = tf.placeholder(tf.float64, [],name="learning_rate")

            if self.for_training:
                feats_batchs = self.prepare_from_base64(self.training_data, for_training=True)
            else:
                feats_batchs = self.prepare_from_base64(self.test_data, for_training=False)
            self.target_ph = feats_batchs[:,0]
            self.uid_batch_ph = feats_batchs[:,1]
            self.mid_batch_ph = feats_batchs[:,2]
            self.cat_batch_ph = feats_batchs[:,3]
            self.noclk_mid_batch_ph = feats_batchs[:,4]
            self.noclk_cat_batch_ph = feats_batchs[:,5]
            self.seq_len_ph = feats_batchs[:,6]
            self.mid_his_batch_ph = feats_batchs[:, 7: 7+self.maxLen]
            self.cat_his_batch_ph = feats_batchs[:, 7+self.maxLen: 7+2*self.maxLen]
            self.mask = numpy.ones((self.batch_size, self.maxLen)).astype('float32')


        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [self.n_uid, self.uid_embedding_dim])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [self.n_mid, self.mid_embedding_dim])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [self.n_cat, self.cat_embedding_dim])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

            self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
            self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
            self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
            if self.use_negsampling:
                self.noclk_item_his_eb = tf.concat(
                    [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)# 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
                self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                    [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])# cat embedding 18 concate item embedding 18.

                self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
                self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
                self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def prepare_from_base64(self, files, for_training=False):
        dataset = tf.data.TextLineDataset(files, compression_type="GZIP")
        dataset = dataset.shuffle(buffer_size=self.batch_size * 500)
        if for_training:
            dataset = dataset.repeat(self.epochs)
        else:
            dataset = dataset.repeat()
        dataset = dataset.map(lambda x: base64_to_int(x), num_parallel_calls=56)
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
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

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
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        loss, accuracy, aux_loss, merged, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.merged, self.optimizer], feed_dict={
            self.for_training: inps[0],
            self.lr: inps[1]
        })
        return loss, accuracy, aux_loss, merged


    def calculate(self, sess, inps):
        probs, targets, uids, loss, accuracy, aux_loss, merged = sess.run([self.y_hat, self.target_ph, self.uid_batch_ph, self.loss, self.accuracy, self.aux_loss, self.merged], feed_dict={
            self.for_training: inps[0],
            self.lr: inps[1]
        }
        return probs, targets, uids, loss, accuracy, aux_loss, merged


    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class DIENModel(BaseModel):
    """DIEN"""
    def __init__(self, conf):
        super(DIENModel, self).__init__(conf)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32, parallel_iterations=256,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

