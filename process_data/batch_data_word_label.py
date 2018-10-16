# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import os
from data_helper import pad_X400_same, train_batch

# w
batch_train_path = '../data/wd_pdQSf400/train/'
batch_valid_path = '../data/wd_pdQSf400/valid/'
batch_test_path = '../data/wd_pdQSf400/test/'


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
if not os.path.exists(batch_train_path):
    os.makedirs(batch_train_path)
if not os.path.exists(batch_valid_path):
    os.makedirs(batch_valid_path)

save_path = '../data/'

# r
file_word2id_te = 'wd_te.npy'
file_word_label_tr = 'word_label_tr.npy'
file_word_label_va = 'word_label_va.npy'

batch_size = 128


def train_get_batch(word_label, batch_path):
    print('loading words and ys.', save_path + word_label)
    word_y = np.load(save_path + word_label)
    words = []
    ys = []
    for word, y in word_y:
        words.append(word)
        ys.append(y)
    words = np.asarray(words)
    ys = np.asarray(ys)
    p = Pool()
    X = np.asarray(list(p.map(pad_X400_same, words)), dtype=np.int64)
    p.close()
    p.join()
    sample_num = X.shape[0]
    np.random.seed(13)
    new_index = np.random.permutation(sample_num)
    X = X[new_index]
    y = ys[new_index]
    train_batch(X, y, batch_path, batch_size)


def valid_get_batch(word_label, batch_path):
    print('loading words and ys.',
          save_path + word_label)
    word_y = np.load(save_path + word_label)
    words = []
    ys = []
    for word, y in word_y:
        # print(word)
        # print(y)
        words.append(word)
        ys.append(y)

    p = Pool()
    X = np.asarray(list(p.map(pad_X400_same, words)), dtype=np.int64)
    p.close()
    p.join()
    train_batch(X, ys, batch_path, batch_size)


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
    valid_get_batch(file_word_label_tr, batch_train_path)  # 644
    train_get_batch(file_word_label_va, batch_valid_path)  # 158
    test_get_batch(file_word2id_te, batch_test_path)  # 801

