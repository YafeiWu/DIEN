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

def config(confpath, date=None):
    with open(confpath, 'r') as f:
        content = f.read()
    paras = yaml.load(content)

    date = "base" if date is None else date
    paras['model_path'] = paras['model_path'].format(date)
    paras['best_model_path'] = paras['best_model_path'].format(date)

    source_dicts = []
    for source_dict in [paras['uid_voc'], paras['mid_voc'], paras['cat_voc']]:
        source_dicts.append(load_voc(source_dict))
    paras['n_uid'] = len(source_dicts[0])
    paras['n_mid'] = len(source_dicts[1])
    paras['n_cat'] = len(source_dicts[2])
    return paras


def eval(sess, model, best_model_path, iter=None):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 1
    stored_arr = []
    while True:
        try:

            prob, target, uids, loss, acc, aux_loss, merged = model.calculate(sess, [False, 0.0])
            loss_sum += loss
            aux_loss_sum = aux_loss
            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            uid_1 = uids.tolist()
            for u, p ,t in zip(uid_1, prob_1, target_1):
                stored_arr.append([u, p, t])
            nums += 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"

    test_auc = cal_auc(stored_arr)
    test_user_auc = cal_user_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if iter is not None and best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, best_model_path+"--"+str(iter))
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
        lr = 0.001
        start_ = time.time()
        for iter in range(conf['max_steps']):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            try:
                loss, acc, aux_loss, train_merged_summary = model.train(sess, [True, lr])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                train_writer.add_summary(train_merged_summary, iter)

                if (iter % conf['test_iter']) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
                                          (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter))
                    test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss, test_merged_summary = eval(sess, model, conf['best_model_path'], iter)
                    test_writer.add_summary(test_merged_summary, iter)
                    print('test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'
                          .format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))

                    print('iter {}. learning rate: {}. take time: {}'.format(iter, lr, time.time()- start_))
                    sys.stdout.flush()
                    start_ = time.time()
                if (iter % 10000) == 0:
                    lr *= 0.5

            except Exception as e:
                print('Exception: {}, Stack: {}'.format(e, traceback.format_exc()))
                sys.exit()

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

