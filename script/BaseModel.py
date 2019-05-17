import tensorflow as tf
import os
from datasource import base64_to_int32

class BaseModel(object):
    def __init__(self, conf, task="train"):
        self.task = task
        self.enable_shuffle = conf['enable_shuffle']

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
                print("INFO -> update_best_model remove #{}".format(os.path.join(save_dir,file_)))
        self.save(sess, path+"--"+str(iter))

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)