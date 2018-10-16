# coding: utf-8

import tensorflow as tf

a = [
    [[2, 1, 3, 5], [6, 4, 3, 9]],
    [[4, 2, 2, 1], [8, 3, 2, 1]],
    [[3, 2, 1, 0], [4, 5, 3, 2]],
    [[3, 2, 2, 1], [8, 5, 3, 7]]
]
a_t = tf.Variable(tf.constant(a))
tf.global_variables_initializer()
sess = tf.Session()

print(sess.run(a_t))