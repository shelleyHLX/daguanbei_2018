# coding: utf-8

import numpy as np

save_path = '../data/'
import random

def get_random_vec():
    words_num = 1280000 + 2
    embed_size = 256
    random.seed(1988)
    words_vectors = np.random.rand(words_num, embed_size)
    np.save(save_path + 'word2vec_' + str(embed_size) + '.npy', words_vectors)
    print('save to', save_path + 'word2vec_' + str(embed_size) + '.npy')
    random.seed(5432)
    chars_vectors = np.random.rand(words_num, embed_size)
    np.save(save_path + 'char2vec' + str(embed_size) + '.npy', chars_vectors)
    print('save to ', save_path + 'char2vec' + str(embed_size) + '.npy')

"""
unk 0
pad 1
"""
if __name__ == '__main__':
    get_random_vec()