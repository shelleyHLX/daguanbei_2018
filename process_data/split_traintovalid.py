# coding: utf-8

import numpy as np

read_dir = '../data/'

train_wd_file = 'wd_tr.npy'
train_ch_file = 'ch_tr.npy'

def split_train():
    num = 20000
    # w
    wd_tr = np.load(read_dir + train_wd_file)
    len_wd = len(wd_tr)
    idx = np.random.permutation(len_wd)
    wd_tr = wd_tr[idx]
    wd_va = wd_tr[0:num]
    wd_tr_s = wd_tr[num:]
    np.save(read_dir + 'wd_va.npy', wd_va)
    print('save to', read_dir + 'wd_va.npy')
    np.save(read_dir + 'wd_tr_s.npy', wd_tr_s)
    print('save to ', read_dir + 'wd_tr_s.npy')
    print('train_wd_file ', len_wd)
    # c
    ch_tr = np.load(read_dir + train_ch_file)
    ch_tr = ch_tr[idx]
    ch_va = ch_tr[0:num]
    ch_tr_s = ch_tr[num:]
    np.save(read_dir + 'ch_va.npy', ch_va)
    print('save to', read_dir + 'ch_va.npy')
    np.save(read_dir + 'ch_tr_s.npy', ch_tr_s)
    print('save to ', read_dir + 'ch_tr_s.npy')
    # label
    label = np.load(read_dir + 'cla_tr.npy')
    label = label[idx]
    label_va = label[0:num]
    label_tr_s = label[num:]
    np.save(read_dir + 'label_va.npy', label_va)
    print('save to ', read_dir + 'label_va.npy')
    np.save(read_dir + 'label_tr_s.npy', label_tr_s)
    print('save to', read_dir + 'label_tr_s.npy')


"""
102277
"""

if __name__ == '__main__':
    split_train()