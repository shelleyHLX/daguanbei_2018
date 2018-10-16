# coding: utf-8

import numpy as np

# score = np.vstack(np.array([
#     [2, 2, 3, 12, 88],
#     [5, 3, 2, 11, 75]
# ]))
#
# print(score)
# print(type(score))
# print(score[0])

# marked_label = [2, 3, 4, 5, 5, 3, 4]
# weights = np.random.uniform(-1, 1, size=len(marked_label))
# print(weights)

s = np.load('best_weights.npy')
print(s[0])