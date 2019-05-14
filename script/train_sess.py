import numpy
import tensorflow as tf
from SeqEmbModel import *
from SessionModel import *
import time
import random
import sys
import traceback
from utils import *
import os

best_auc = 0.

def eval(sess, model, best_model_path, iter=None, test_batches=None):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []
    nums = 1
    while not test_batches or nums < test_batches:
        try:
            prob, target, uids, loss, aux_loss, top1_acc, acc = model.calculate(sess, [False, 0.0])
            loss_sum += loss
            aux_loss_sum = aux_loss
            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()
            uid_1 = uids.tolist()
            for u, p ,t in zip(uid_1, prob_1, target_1):
                stored_arr.append([u, p, t])
            nums += 1
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
    print("INFO -> Eval test_auc #{}, best_auc #{}".format(test_auc, best_auc))
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
        if conf['model_type'] == 'SEMB':
            model = SeqEmbModel(conf)
        elif conf['model_type'] == 'SESS':
            model = SessionModel(conf)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tf.summary.FileWriter(conf['logdir'], sess.graph)
        print("local_variables_initializer done")

        start_first = time.time()
        lr = conf['learning_rate']
        loss_sum,  aux_loss_sum, top1_acc_sum, target_acc_sum,= 0., 0., 0., 0.
        test_loss_sum, test_aux_loss_sum, test_top1_acc_sum, test_target_acc_sum, = 0., 0., 0., 0.
        test_loss, test_aux_loss, test_top1_acc, test_target_acc, test_merged_summary = model.test(sess, [False, lr])
        print('iter: %d ----> test_loss: %.4f ---- test_aux_loss: %.4f ---- test_top1_accuracy: %.4f ---- test_target_accuracy: %.4f' % \
              (0, test_loss, test_aux_loss, test_top1_acc, test_target_acc))
        sys.stdout.flush()

        start_ = time.time()
        for iter in range(1,conf['max_steps']):
            try:
                loss, aux_loss, top1_acc, target_acc, train_merged_summary = model.train(sess, [True, lr])
                loss_sum += loss
                aux_loss_sum += aux_loss
                top1_acc_sum += top1_acc
                target_acc_sum += target_acc
                train_writer.add_summary(train_merged_summary, iter)

                if iter % test_iter == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_aux_loss: %.4f ---- train_top1_accuracy: %.4f ---- train_target_accuracy: %.4f' % \
                          (iter, loss_sum / test_iter, aux_loss_sum / test_iter, top1_acc_sum / test_iter, target_acc_sum / test_iter))

                    test_loss, test_aux_loss, test_top1_acc, test_target_acc, test_merged_summary = model.test(sess, [False, lr])
                    test_loss_sum += test_loss
                    test_aux_loss_sum += test_aux_loss
                    test_top1_acc_sum += test_top1_acc
                    test_target_acc_sum += test_target_acc
                    test_writer.add_summary(test_merged_summary, iter)

                    print('iter: %d ----> test_loss: %.4f ---- test_aux_loss: %.4f ---- test_top1_accuracy: %.4f ---- test_target_accuracy: %.4f' % \
                          (iter, test_loss_sum / test_iter, test_aux_loss_sum / test_iter, test_top1_acc_sum / test_iter, test_target_acc_sum / test_iter))

                    sys.stdout.flush()
                    loss_sum,  aux_loss_sum, top1_acc_sum, target_acc_sum,= 0., 0., 0., 0.
                    test_loss_sum, test_aux_loss_sum, test_top1_acc_sum, test_target_acc_sum, = 0., 0., 0., 0.

                if iter % conf['save_iter'] == 0:

                    test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, iter=iter, test_batches=conf['save_iter'])
                    print('iter: %d ----> test_loss: %.4f ---- test_aux_loss: %.4f ---- test_accuracy: %.4f ---- test_auc: %.4f ---- test_user_auc: %.4f' % \
                          (iter, test_loss, test_aux_loss, test_accuracy, test_auc, test_user_auc))

                    print('iter: {} ----> learning rate: {}. {} iters take time: {}'.format(iter, lr, test_iter, time.time()- start_))
                    start_ = time.time()

                if (iter % conf['lr_decay_steps']) == 0:
                    lr *= 0.5

            except Exception as e:
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
        if conf['model_type'] == 'SEMB':
            model = SeqEmbModel(conf, task='test')
        elif conf['model_type'] == 'SESS':
            model = SessionModel(conf, task='test')
        latest_model = tf.train.latest_checkpoint(model_dir)
        model.restore(sess, latest_model)
        #### test_batches=100 test for all, get per_user_auc
        test_auc, test_user_auc, test_loss, test_accuracy, test_aux_loss = eval(sess, model, best_model_path, iter=None, test_batches=None)
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


