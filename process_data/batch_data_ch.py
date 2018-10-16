# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import os
from data_helper import pad_X400_same, train_batch

# w
batch_train_path = '../data/ch_pdQS400/train/'
batch_valid_path = '../data/ch_pdQS400/valid/'
batch_test_path = '../data/ch_pdQS400/test/'


def test_batch(X, batch_path, batch_size=128):
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    sample_num = len(X)
    batch_num = 0
    for start in list(range(0, sample_num, batch_size)):
        print(batch_num)
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npz'
        X_batch = X[start:end]
        np.savez(batch_name, X=X_batch)
        batch_num += 1
    print('Finished, batch_num=%d' % (batch_num + 1))


if not os.path.exists(batch_test_path):
    os.makedirs(batch_test_path)
# if not os.path.exists(batch_train_path):
#     os.makedirs(batch_train_path)
# if not os.path.exists(batch_valid_path):
#     os.makedirs(batch_valid_path)

save_path = '../data/'

# r
file_word2id_te = 'wd_te.npy'
file_word2id_tr = 'wd_tr_s.npy'
file_word2id_va = 'wd_va.npy'

y_file_train = 'label_tr_s.npy'
y_file_valid = 'label_va.npy'

batch_size = 128

# def train_get_batch(wd_fact2id_path, y_path, batch_path):
#     print('loading facts and ys.',
#           save_path + wd_fact2id_path,
#           save_path + y_path)
#     facts = np.load(save_path + wd_fact2id_path)
#     y = np.load(save_path + y_path)
#     p = Pool()
#     X = np.asarray(list(p.map(pad_X400_same, facts)), dtype=np.int64)
#     p.close()
#     p.join()
#     sample_num = X.shape[0]
#     np.random.seed(13)
#     new_index = np.random.permutation(sample_num)
#     X = X[new_index]
#     y = y[new_index]
#     train_batch(X, y, batch_path, batch_size)
#
# def valid_get_batch(wd_fact2id_path, y_path, batch_path):
#     print('loading facts and ys.',
#           save_path + wd_fact2id_path,
#           save_path + y_path)
#     facts = np.load(save_path + wd_fact2id_path)
#     y = np.load(save_path + y_path)
#     p = Pool()
#     X = np.asarray(list(p.map(pad_X400_same, facts)), dtype=np.int64)
#     p.close()
#     p.join()
#     train_batch(X, y, batch_path, batch_size)


def test_get_batch(wd_fact2id_path, batch_path):
    print('loading facts and ys.',
          save_path + wd_fact2id_path)
    facts = np.load(save_path + wd_fact2id_path)
    print(len(facts))
    p = Pool()
    X = np.asarray(list(p.map(pad_X400_same, facts)), dtype=np.int64)
    p.close()
    p.join()
    test_batch(X, batch_path, batch_size)


if __name__ == '__main__':
    # valid_get_batch(file_word2id_va, y_file_valid, batch_valid_path)
    # train_get_batch(file_word2id_tr, y_file_train, batch_train_path)
    test_get_batch(file_word2id_te, batch_test_path)
