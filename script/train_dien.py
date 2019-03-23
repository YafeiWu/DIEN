import numpy
import tensorflow as tf
from DienModel import *
import time
import random
import sys
import traceback
from utils import *
import os

best_auc = 0.

def eval(sess, model, best_model_path, iter=None, test_batches=1):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []
    for nums in range(1,test_batches+1):
        try:
            prob, target, uids, loss, acc, aux_loss = model.calculate(sess, [False, 0.0])
            loss_sum += loss
            aux_loss_sum = aux_loss
            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            uid_1 = uids.tolist()
            for u, p ,t in zip(uid_1, prob_1, target_1):
                stored_arr.append([u, p, t])
        except Exception as e :
            print("eval Error : {}".format(traceback.format_exc(e)))
            print("End of test dataset")  # ==> "End of test dataset"
            break

    test_auc = cal_auc(stored_arr)
    test_user_auc = cal_user_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum = aux_loss_sum / nums
    global best_auc
    if iter is not None and best_auc < test_auc:
        model.update_best_model(sess, best_model_path, iter)
        best_auc = test_auc
    return test_auc, test_user_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(conf, seed):
    best_model_path = conf['best_model_path'] + str(seed)
    train_writer = tf.summary.FileWriter("{}/train".format(conf['logdir']))
    test_writer = tf.summary.FileWriter("{}/test".format(conf['logdir']))
    test_iter = conf['test_iter']
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    with tf.Session(config=session_config) as sess:
        model = DIENModel(conf)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.summary.FileWriter(conf['logdir'], sess.graph)
        print("local_variables_initializer done")

        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, iter=None, test_batches=1)
        print('iter: %d ----> test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % \
              (0, test_loss, test_accuracy, test_aux_loss))
        print('iter: {} ----> test_auc: {} ---- test_user_auc: {} '.format(0, test_auc, test_user_auc))
        sys.stdout.flush()

        start_first = time.time()
        lr = 0.001
        loss_sum, accuracy_sum, aux_loss_sum= 0., 0., 0.
        test_loss_sum, test_accuracy_sum, test_aux_loss_sum= 0., 0., 0.
        start_ = time.time()
        for iter in range(1,conf['max_steps']):
            try:
                loss, acc, aux_loss, train_merged_summary = model.train(sess, [True, lr])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                train_writer.add_summary(train_merged_summary, iter)

                test_loss, test_accuracy, test_aux_loss, test_merged_summary = model.test(sess, [False, lr])
                test_loss_sum += test_loss
                test_accuracy_sum += test_accuracy
                test_aux_loss_sum += test_aux_loss
                test_writer.add_summary(test_merged_summary, iter)

                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                                          (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))

                    # test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, iter, test_batches=100)
                    print('iter: %d ----> test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % \
                          (iter, test_loss_sum / test_iter, test_accuracy_sum / test_iter, test_aux_loss_sum / test_iter))

                    # print('iter: {} ----> test_auc: {} ---- test_user_auc: {} '.format(iter, test_auc, test_user_auc))
                    print('iter: {} ----> learning rate: {}. {} iters take time: {}'.format(iter, lr, test_iter, time.time()- start_))

                    sys.stdout.flush()
                    loss_sum, accuracy_sum, aux_loss_sum= 0., 0., 0.
                    start_ = time.time()

                if (iter % conf['lr_decay_steps']) == 0:
                    lr *= 0.5

            except Exception as e:
                print("training Error : {}".format(traceback.format_exc(e)))
                print("End of training dataset")  # ==> "End of training dataset"
                break

        # #### test_batches=100 test for all, get per_user_auc
        # test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, None, 100)
        # print('All Test Users. test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'
        #       .format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))
        print('Training done. Take time:{}'.format(time.time()-start_first))

def test(conf, seed):
    best_model_path = conf['best_model_path'] + str(seed)
    model_dir , prefix = os.path.split(best_model_path)
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    with tf.Session(config=session_config) as sess:
        model = DIENModel(conf, task="test")
        latest_model = tf.train.latest_checkpoint(model_dir)
        model.restore(sess, latest_model)
        #### test_batches=100 test for all, get per_user_auc
        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, None, 100)
        print('All Test Users. test_auc: {} ---- test_user_auc: {} ---- test_loss: {} ---- test_accuracy: {} ---- test_aux_loss: {}'
              .format(test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss))

if __name__ == '__main__':
    conf = config(sys.argv[2], sys.argv[3])
    print("--------------- Model Config --------------- \n{}\n".format(conf))
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


