# coding: utf-8

import numpy as np

save_path = '../data/'
# r
word_tr = 'wd_tr_filter.npy'
word_te = 'wd_te_filter.npy'
word_va = 'wd_va_filter.npy'
y_tr = 'label_tr_s.npy'
y_va = 'label_va.npy'

# w
word_label_te = 'word_label_te.npy'
word_label_tr = 'word_label_tr.npy'
word_label_va = 'word_label_va.npy'

def read_words_labels(word_file, y_file, write_file):
    words = np.load(save_path + word_file)
    y = np.load(save_path + y_file)
    length = len(y)
    word_label = []
    print(y[0])
    print(words[0])
    i = 0
    for i in range(length):
        if len(words) > 10:
            word_label.append((words[i], y[i][0]))
            i += 1
    np.save(save_path + write_file, word_label)
    print(word_label[0])
    print('save to ', save_path + write_file)
    print('iiiiiiiiiiiiiii', i)

if __name__ == '__main__':
    read_words_labels(word_tr, y_tr, word_label_tr)  # 82277
    read_words_labels(word_va, y_va, word_label_va)  # 20000
