# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import network
from utils import get_logger
import codecs

sys.path.append('../..')
from evaluator import daguanbei_predict, cail_evaluator

settings = network.Settings()
model_name = settings.model_name
ckpt_path = settings.ckpt_path

scores_path = '../../scores/'

if not os.path.exists(scores_path):
    os.makedirs(scores_path)

# embedding_path = '../../data/dealt_word_embedding.npy'
embedding_path = '../../data/word2vec.npy'

data_valid_path = '../../data/wd_pdQS400/valid/'
data_test_path = '../../data/wd_pdQS400/test/'

va_batches = os.listdir(data_valid_path)
te_batches = os.listdir(data_test_path)  # batch 文件名列表
n_va_batches = len(va_batches)
n_te_batches = len(te_batches)


def get_batch(data_path, batch_id):
    """get a batch from data_path"""
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    y_batch = new_batch['y']
    return [X_batch, y_batch]


def get_test_batch(data_path, batch_id):
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    return X_batch


def predict(sess, model, logger):
    """Test on the test data."""
    te_batches = os.listdir(data_test_path)
    n_te_batches = len(te_batches)
    predict_labels_list = list()  # 所有的预测结果
    for i in tqdm(range(n_te_batches)):
        X_batch = get_test_batch(data_test_path, i)
        _batch_size = len(X_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs: X_batch,
                     model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels_list.extend(predict_labels)
    predict_scores_file = scores_path + model_name + '/' + 'predict.npy'
    np.save(predict_scores_file, predict_labels_list)
    labels = daguanbei_predict(predict_labels_list)
    f_w = codecs.open(scores_path + model_name + '/' + 'predict.csv', 'a', 'utf-8')
    f_w.write('id,class\n')
    i = 0
    for label in labels:
        f_w.write(str(i) + ',' + str(label) + '\n')
        i = i + 1
    f_w.close()


def predict_valid(sess, model, logger):
    """Test on the valid data."""
    time0 = time.time()
    predict_labels_list = list()  # 所有的预测结果
    marked_labels_list = list()
    for i in tqdm(range(int(n_va_batches))):
        [X_batch, y_batch] = get_batch(data_valid_path, i)
        marked_labels_list.extend(y_batch)
        _batch_size = len(X_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X_inputs: X_batch,
                     model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_labels_list.extend(predict_labels)

    f1_micro, f1_macro, score12 = cail_evaluator(predict_labels_list, marked_labels_list)
    print('precision_micro=%g, recall_micro=%g, score12=%g, time=%g s'
          % (f1_micro, f1_macro, score12, time.time() - time0))
    logger.info('\nValid predicting...\nEND:Global_step={}: f1_micro={}, f1_macro={}, score12={}, time=%g s'.
                format(sess.run(model.global_step), f1_micro, f1_macro, score12, time.time() - time0))


def main(_):
    if not os.path.exists(ckpt_path + 'checkpoint'):
        print('there is not saved model, please check the ckpt path')
        exit()
    print('Loading model...')
    W_embedding = np.load(embedding_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_path = scores_path + settings.model_name + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = get_logger(log_path + 'predict.log')
    with tf.Session(config=config) as sess:
        model = network.TextCNN(W_embedding, settings)
        # ckpt_path: /ckpt/wd_BiGRU/checkpoint
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        print('Valid predicting...')
        predict_valid(sess, model, logger)

        print('Test predicting...')
        predict(sess, model, logger)


if __name__ == '__main__':
    tf.app.run()

