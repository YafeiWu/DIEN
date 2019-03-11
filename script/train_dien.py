import numpy
from data_source import DataSource
import tensorflow as tf
from DienModel import *
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


def eval(sess, test_data, model, model_path, iter=None):

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
    if best_auc < test_auc and iter is not None:
        best_auc = test_auc
        model.save(sess, model_path+"--"+str(iter))
    return test_auc, test_user_auc, loss_sum, accuracy_sum, aux_loss_sum, merged

def train(conf, seed):
    best_model_path = conf['best_model_path'] + str(seed)
    train_writer = tf.summary.FileWriter("{}/train".format(conf['logdir']))
    test_writer = tf.summary.FileWriter("{}/test".format(conf['logdir']))
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = DIENModel(conf)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("local_variables_initializer done")
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        for epoch in range(conf['epochs']):
            start_one_epoch = time.time()
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                try:
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
                        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss, test_merged_summary = eval(sess, test_data, model, best_model_path, iter)
                        print('test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'.format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))
                        test_writer.add_summary(test_merged_summary, iter)
                        loss_sum = 0.0
                        accuracy_sum = 0.0
                        aux_loss_sum = 0.0
                    # if (iter % save_iter) == 0:
                    #     print('save model iter: %d' %(iter))
                    #     model.save(sess, model_path+"--"+str(iter))
                except Exception as e:
                    print('Exception: {}, Stack: {}'.format(e, traceback.format_exc()))
                    sys.exit()
            print('epoch {}. learning rate: {}. take time: {}'.format(epoch, lr, time.time()-start_one_epoch))
            lr *= 0.5
        print('training done. take time:{}'.format(time.time()-start_time))

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


