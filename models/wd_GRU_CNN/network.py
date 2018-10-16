# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

"""
wd_BiGRU_Attention
bigru + attention
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Settings(object):
    def __init__(self):
        # self.model_name = 'wd_BiGRU'
        self.model_name = 'wd_GRU_CNN_pdSQ1000'
        self.fact_len = 1000
        # CNN
        self.filter_sizes = [2, 3, 4]
        self.n_filter = 256
        # gru
        self.gru_hidden_size = 128
        self.n_layer = 1
        # FC
        self.fc_hidden_size = 1024
        self.n_class = 19
        # write
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class GRU_CNN(object):
    """
    fact: inputs -> bigru+cnn -> output
    output -> fc+bn+relu -> sigmoid_entropy.
    """
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        # CNN
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)  # 256 * [2, 3, 4, 5 ,7]
        self.gru_hidden_size = settings.gru_hidden_size
        self.n_layer = settings.n_layer
        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int64, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding),
                                             trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('BiGRU'):
            output = self.bigru_cnn_inference(self._X_inputs)

        with tf.variable_scope('fc-bn-layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')  # 每个类别的分数 scores

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
            tf.summary.scalar('loss', self._loss)

        self.saver = tf.train.Saver(max_to_keep=1)

    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X_inputs(self):
        return self._X_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batch normalization.
        Args:
            Ylogits: 1D向量或者是3D的卷积结果。
            num_updates: 迭代的global_step
            offset：表示beta，全局均值；在 RELU 激活中一般初始化为 0.1。
            scale：表示lambda，全局方差；在 sigmoid 激活中需要，这 RELU 激活中作用不大。
            m: 表示batch均值；v:表示batch方差。
            bnepsilon：一个很小的浮点数，防止除以 0.
        Returns:
            Ybn: 和 Ylogits 的维度一样，就是经过 Batch Normalization 处理的结果。
            update_moving_everages：更新mean和variance，主要是给最后的 test 使用。
        """
        print('================= batch norm ================')
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, self._global_step)
        # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def gru_cell(self):
        print('============= gru_cell =============')
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.gru_hidden_size, reuse=tf.get_variable_scope().reuse)
            print('type(cell): ', type(cell))
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs):
        print('================== bi_gru ==================')
        """build the Bi-GRU network. 返回个所有层的隐含状态。"""
        cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        print('inputs: ', inputs.shape)  # (?, 200, 256)
        print('cells_fw: ', type(cells_fw))
        print('cells_bw: ', type(cells_bw))
        print('celss_fw: ', np.asarray(cells_fw).shape)
        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        print('initial_states_bw: ', np.asarray(initial_states_bw).shape)
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw,
                                                            dtype=tf.float32)
        print('outputs: ', outputs.shape)  # (?, 200, self.hidden_size * 2)

        return outputs

    def cnn_inference(self, inputs, n_step):  # fact_len = n_step
        """TextCNN 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        print('======================into cnn_inference======================')
        # print('inputs.shape: ', inputs.shape)  # (?, 200, 256)
        inputs = tf.expand_dims(inputs, -1)
        print('inputs.shape: ', inputs.shape)  # (?, 200, 256, 1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size) # 2，3,4,5,7
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                # print('filter_shape: ', np.asarray(filter_shape))

                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                # print('W_filter shape: ', W_filter)

                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')
                # print('beta: ', beta.shape)  #  (256,)(256,) (256,)
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # print('conv: ', conv.shape)

                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                # print('h: ', h.shape)

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # print('pooled: ', pooled.shape)

                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        # print('pooled_outputs: ', np.asarray(pooled_outputs).shape) #  (5,)
        h_pool = tf.concat(pooled_outputs, 3)
        # print('h_pool.shape', h_pool.shape)  # (?, 1, 1, 1280) 1280 = 256 × 5
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        # print('h_pool_flat.shape', h_pool_flat.shape)  # (?, 1280)
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]

    def bigru_cnn_inference(self, X_inputs):
        print('============ bigru_inference =============')
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        print('X_inputs: ', X_inputs.shape)
        print('inputs: ', inputs.shape)
        output_bigru = self.bi_gru(inputs)
        print('output_bigru: ', output_bigru.shape)  # (?, 200, self.hidden_size*2)
        output_cnn = self.cnn_inference(output_bigru, self.fact_len)
        print('output_cnn: ', output_cnn.shape) # (?, self.hidden_size * 2)
        return output_cnn

"""
============ bigru_inference =============
X_inputs:  (?, 200)
inputs:  (?, 200, 256)
================== bi_gru ==================
============= gru_cell =============
type(cell):  <class 'tensorflow.python.ops.rnn_cell_impl.GRUCell'>
============= gru_cell =============
type(cell):  <class 'tensorflow.python.ops.rnn_cell_impl.GRUCell'>
inputs:  (?, 200, 256)
cells_fw:  <class 'list'>
cells_bw:  <class 'list'>
celss_fw:  (1,)
initial_states_bw:  (1,)
outputs:  (?, 200, 512)
output_bigru:  (?, 200, 512)
======================into cnn_inference======================
inputs.shape:  (?, 200, 512, 1)
conv-maxpool-2
================= batch norm ================
conv-maxpool-3
================= batch norm ================
conv-maxpool-4
================= batch norm ================
output_cnn:  (?, 768)
================= batch norm ================
"""

# test the model
def Bi_GRU_test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    W_embedding = np.random.randn(50, 256)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = BiGRU(W_embedding, settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.loss)
        update_op = tf.group(*model.update_emas)
        sess.run(tf.global_variables_initializer())
        fetch = [model.loss, model.y_pred, train_op, update_op]
        loss_list = list()
        for i in range(100):
            X_batch = np.zeros((batch_size, 200), dtype=float)
            y_batch = np.zeros((batch_size, 183), dtype=int)
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.tst: False, model.keep_prob: 0.5}
            loss, y_pred, _, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss)

if __name__ == '__main__':
    Bi_GRU_test()
