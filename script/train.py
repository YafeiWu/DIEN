import numpy
from data_iterator import DataIterator
from data_iterator_v2 import DataIteratorV2
import tensorflow as tf
from model import *
import time
import random
import sys
import traceback
from utils import *
import yaml

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.

def config(configpath):
    with open(configpath, 'r') as f:
        content = f.read()
    paras = yaml.load(content)
    # if date is not None:
    #     self.paras['modeldir'] = self.paras['modeldir'].format(date)
    #     self.paras['logdir'] = self.paras['logdir'].format(date)
    return paras

def prepare_data(input, target, maxlen = None, return_neg = False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = numpy.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        # print('DEBUG prepare_data: uids shape:{}, mids shape:{}, mid_his shape:{}, cat_his shape:{}, noclk_cat_his shape:{}'.format(
        #                             uids.shape, mids.shape, mid_his.shape, cat_his.shape, noclk_cat_his.shape))
        # print('DEBUG prepare_data: lengths_x len: {}'.format(lengths_x))
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)

def eval(sess, test_data, model, model_path):

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        nums += 1
        prob, loss, acc, aux_loss, merged = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        uid_1 = uids.tolist()
        for u, p ,t in zip(uid_1, prob_1, target_1):
            stored_arr.append([u, p, t])
    test_auc = cal_auc(stored_arr)
    test_user_auc = cal_user_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, test_user_auc, loss_sum, accuracy_sum, aux_loss_sum, merged

def train(conf,seed):
    train_file = conf['train_file']
    test_file = conf['test_file']
    uid_voc = conf['uid_voc']
    mid_voc = conf['mid_voc']
    cat_voc = conf['cat_voc']
    item_info = conf['item_info']
    batch_size = conf['batch_size']
    maxlen = conf['maxlen']
    minlen  = conf['minlen']
    model_type = conf['model_type']
    test_iter = conf['test_iter']
    save_iter = conf['save_iter']
    epochs = conf['epochs']
    enable_shuffle = conf['enable_shuffle']

    model_path = conf['model_path'] + model_type + str(seed)
    best_model_path = conf['best_model_path'] + model_type + str(seed)
    train_writer = tf.summary.FileWriter("{}/train".format(conf['logdir/train']))
    test_writer = tf.summary.FileWriter("{}/test".format(conf['logdir']))
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIteratorV2(train_file, uid_voc, mid_voc, cat_voc, item_info, batch_size, maxlen, minlen=minlen, shuffle_each_epoch=enable_shuffle)
        train_data.print_data_info()
        test_data = DataIteratorV2(test_file, uid_voc, mid_voc, cat_voc, item_info, batch_size, maxlen, minlen=minlen, shuffle_each_epoch=False)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        print("{} build".format(model_type))
        # model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("{} local_variables_initializer done".format(model_type))
        sys.stdout.flush()
        # test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss, test_merged_summary = eval(sess, test_data, model, best_model_path)
        # print('test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'.format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))
        # sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for epoch in range(epochs):
            start_one_epoch = time.time()
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                try:
                    uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
                    loss, acc, aux_loss, train_merged_summary = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats])
                    loss_sum += loss
                    accuracy_sum += acc
                    aux_loss_sum += aux_loss
                    iter += 1
                    train_writer.add_summary(train_merged_summary, iter)
                    sys.stdout.flush()
                    if (iter % test_iter) == 0:
                        print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                                              (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss, test_merged_summary = eval(sess, test_data, model, best_model_path)
                        print('test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'.format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))
                        test_writer.add_summary(test_merged_summary, iter)
                        loss_sum = 0.0
                        accuracy_sum = 0.0
                        aux_loss_sum = 0.0
                    if (iter % save_iter) == 0:
                        print('save model iter: %d' %(iter))
                        model.save(sess, model_path+"--"+str(iter))
                except Exception as e:
                    print('Exception: {}, Stack: {}'.format(e, traceback.format_exc()))
                    sys.exit()
            print('epoch {}. learning rate: {}. take time: {}'.format(epoch, lr, time.time()-start_one_epoch))
            lr *= 0.5
        print('training done. take time:{}'.format(time.time()-start_time))

def test(conf, seed):
    train_file = conf['train_file']
    test_file = conf['test_file']
    uid_voc = conf['uid_voc']
    mid_voc = conf['mid_voc']
    cat_voc = conf['cat_voc']
    item_info = conf['item_info']
    batch_size = conf['batch_size']
    maxlen = conf['maxlen']
    minlen  = conf['minlen']
    model_type = conf['model_type']
    enable_shuffle = conf['enable_shuffle']

    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIteratorV2(train_file, uid_voc, mid_voc, cat_voc, item_info, batch_size, maxlen, minlen=minlen, shuffle_each_epoch=False)
        train_data.print_data_info()
        test_data = DataIteratorV2(test_file, uid_voc, mid_voc, cat_voc, item_info, batch_size, maxlen, minlen=minlen, shuffle_each_epoch=False)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-att-gru':
            model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-gru-att':
            model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-qa-attGru':
            model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN-V2-gru-vec-attGru':
            model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print ("Invalid model_type : %s", model_type)
            return
        model.restore(sess, model_path)
        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss, test_merged_summary = eval(sess, test_data, model, model_path)
        print('test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'.format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))

if __name__ == '__main__':
    conf = config(sys.argv[2])
    print("Model Config : {}".format(conf))
    SEED = conf['seed']
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    if sys.argv[1] == 'train':
        train(conf, seed=SEED)
    elif sys.argv[1] == 'test':
        test(conf, seed=SEED)
    else:
        print('do nothing...')


