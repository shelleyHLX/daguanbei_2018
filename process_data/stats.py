# coding: utf-8

import codecs
import numpy as np

write_dir = '../data/'
read_dir = '../new_data/'
# w
char_data = 'char_tr.npy'
word_data = 'word_tr.npy'

# r
train_data = 'train_set.csv'
test_data = 'test_set.csv'


def train_split_word_char():
    c_data = codecs.open(read_dir + train_data, 'r', 'utf-8')
    line = c_data.readline()
    max_word = -1
    min_word = 10000
    max_char = -1
    min_char = 10000
    max_cla = -1
    min_cla = 10000
    cls_all = []
    chars_all = []
    words_all = []
    for line in c_data.readlines():
        id, article, word_seg, cla = line.split(',')
        chars = article.split()
        words = word_seg.split()
        clas = cla.split()
        cla_int = []
        for cl in clas:
            cla_int.append(int(cl))
        cls_all.append(cla_int)
        char_int = []
        for ch in chars:
            char_int.append(int(ch))
        chars_all.append(char_int)
        word_int = []
        for wd in words:
            word_int.append(int(wd))
        words_all.append(word_int)

    for c in cls_all:
        if max_cla < max(c):
            max_cla = max(c)
        if min_cla > min(c):
            min_cla = min(c)

    for ch in chars_all:
        if max_char < max(ch):
            max_char = max(ch)
        if min_char > min(ch):
            min_char = min(ch)

    for wd in words_all:
        if max_word < max(wd):
            max_word = max(wd)
        if min_word > min(wd):
            min_word = min(wd)
    np.save(write_dir + 'cla.npy', cls_all)
    print('save to ', write_dir + 'cla.npy')
    np.save(write_dir + 'ch.npy', chars_all)
    print('save to ', write_dir + 'ch.npy')
    np.save(write_dir + 'wd.npy', words_all)
    print('save to ', write_dir + 'wd.npy')
    print(max_word, min_word)
    print(max_char, min_char)
    print(max_cla, min_cla)
"""
1279999 0
1279954 13
19 1
"""
def test_split_word_char():
    c_data = codecs.open(read_dir + test_data, 'r', 'utf-8')
    line = c_data.readline()
    max_word = -1
    min_word = 10000
    max_char = -1
    min_char = 10000
    chars_all = []
    words_all = []
    for line in c_data.readlines():
        id, article, word_seg = line.split(',')
        chars = article.split()
        words = word_seg.split()
        char_int = []
        for ch in chars:
            char_int.append(int(ch))
        chars_all.append(char_int)
        word_int = []
        for wd in words:
            word_int.append(int(wd))
        words_all.append(word_int)

    for ch in chars_all:
        if max_char < max(ch):
            max_char = max(ch)
        if min_char > min(ch):
            min_char = min(ch)

    for wd in words_all:
        if max_word < max(wd):
            max_word = max(wd)
        if min_word > min(wd):
            min_word = min(wd)
    np.save(write_dir + 'ch_te.npy', chars_all)
    print('save to ', write_dir + 'ch_te.npy')
    np.save(write_dir + 'wd_te.npy', words_all)
    print('save to ', write_dir + 'wd_te.npy')
    print(max_word, min_word)
    print(max_char, min_char)
"""
1279998 0
1279989 13
"""

if __name__ == '__main__':
    # train_split_word_char()
    test_split_word_char()