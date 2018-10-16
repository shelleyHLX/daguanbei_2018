# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
"""
TextCNN
"""

class Settings(object):
    def __init__(self):
        # self.model_name = 'wd_TextCNN_200'
        self.model_name = 'TextCNNPool_pdQS1000'

        self.fact_len = 800  # 词的个数，行数

        self.filter_sizes = [2, 3, 4]
        self.n_filter = 128
        self.fc_hidden_size = 256

        self.n_class = 19
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'

class TextCNN(object):
    """
    fact: inputs->textcnn->output
    output -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len

        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)  # 256 * [2, 3, 4, 5 ,7]

        self.n_class = settings.n_class
        self.fc_hidden_size = settings.fc_hidden_size

        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            print('===============Inputs===============')
            self._X_inputs = tf.placeholder(tf.int64, [None, self.fact_len], name='X_inputs')
            # print('self._X_inputs.shape', self._X_inputs.shape) # (?, 200)
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_inputs')
            # print('self._y_inputs', self._y_inputs.shape)  # (?, 183)
            """
            self._X_inputs.shape (?, 200)
self._y_inputs (?, 183)
            """

        with tf.variable_scope('embedding'):
            print(print('===============embedding==============='))
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
            print('self.embedding', self.embedding.shape)  # (114394, 256)
        self.embedding_size = W_embedding.shape[1]
        print('self.embedding_size', self.embedding_size)  # 256
        """
        None
self.embedding (114394, 256)
self.embedding_size 256
        """

        with tf.variable_scope('TextCNN'):
            print('===============TextCNN===============')
            output = self.cnn_inference(self._X_inputs, self.fact_len)
            print('output: ', output.shape)  # output:  (?, 1280)

        with tf.variable_scope('fc-bn-layer'):
            print('===============fc-bn-layer===============')
            W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
            print('W_fc.shape: ', W_fc.shape)  # (1280, 1024)
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            print('h_fc', h_fc.shape)  # (?, 1024)
            # beta_fc：一个实数
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            print('beta_fc', beta_fc.shape)  # (1024,)
            tf.summary.histogram('beta_fc', beta_fc)
            # h_fc：矩阵
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu, self.keep_prob)
        """
        W_fc.shape:  (1280, 1024)
h_fc (?, 1024)
beta_fc (1024,)
        """

        with tf.variable_scope('out_layer'):
            print('===============out_layer===============')
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out') # (1024, 183)
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')  # (183,)
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')  # 每个类别的分数 scores

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
            tf.summary.scalar('loss', self._loss)

        self.saver = tf.train.Saver(max_to_keep=2)

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
        """batch-normalization.
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
        # adding the iteration prevents from averaging across non-existing iterations
        print('======================batchnorm======================')
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)
        # decay是衰减率: 0.999,
        bnepsilon = 1e-5
        # print('Ylogtis: ', Ylogits.shape)
        #  (?, 199, 1, 256)
        # (?, 198, 1, 256)
        # (?, 197, 1, 256)
        # (?, 196, 1, 256)
        # (?, 194, 1, 256)
        # print('offset: ', offset.shape)
        #  (256,)
        # (256,)
        # (256,)
        # (256,)
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def cnn_inference(self, X_inputs, n_step):  # fact_len = n_step
        """TextCNN 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        print('======================into cnn_inference======================')
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        # print('inputs.shape: ', inputs.shape)  # (?, 200, 256)
        inputs = tf.expand_dims(inputs, -1)
        print('inputs.shape: ', inputs.shape)  # (?, 200, 256, 1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)  # 2，3,4,5,7
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                print('filter_shape: ', np.asarray(filter_shape))
                #  [  2 256   1 256]
                # [  3 256   1 256]
                # [  4 256   1 256]
                # [  5 256   1 256]
                # [  7 256   1 256]
                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                print('W_filter shape: ', W_filter)
                #  (2, 256, 1, 256)
                # (3, 256, 1, 256)
                # (4, 256, 1, 256)
                # (7, 256, 1, 256)
                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')
                # print('beta: ', beta.shape)  #  (256,)(256,) (256,)
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                print('conv: ', conv.shape)
                #  (?, 199, 1, 256)
                # (?, 198, 1, 256)
                # (?, 197, 1, 256)
                # (?, 196, 1, 256)
                # (?, 196, 1, 256)
                # (?, 194, 1, 256)
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                print('h: ', h.shape)
                #  (?, 199, 1, 256)
                # (?, 198, 1, 256)
                # (?, 197, 1, 256)
                # (?, 196, 1, 256)
                # (?, 194, 1, 256)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                print('pooled: ', pooled.shape)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                #  (?, 1, 1, 256)
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        print('pooled_outputs: ', np.asarray(pooled_outputs).shape)  # (5,)
        h_pool = tf.concat(pooled_outputs, 3)
        print('h_pool.shape', h_pool.shape)  # (?, 1, 1, 1280) 1280 = 256 × 5
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        print('h_pool_flat.shape', h_pool_flat.shape)  # (?, 1280)
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]
    """
    inputs.shape:  (?, 200, 256)
inputs.shape:  (?, 200, 256, 1)
conv-maxpool-2
filter_shape:  [  2 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-2/W_filter:0' shape=(2, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 199, 1, 256)
    """

# test the model
def TextCNN_test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    W_embedding = np.random.randn(50, 256)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = TextCNN(W_embedding, settings)
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
    TextCNN_test()

"""
===============Inputs===============
self._X_inputs.shape (?, 200)
self._y_inputs (?, 183)
===============embedding===============
None
self.embedding (114394, 256)
self.embedding_size 256
===============TextCNN===============
======================into cnn_inference======================
inputs.shape:  (?, 200, 256)
inputs.shape:  (?, 200, 256, 1)
conv-maxpool-2
filter_shape:  [  2 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-2/W_filter:0' shape=(2, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 199, 1, 256)
======================batchnorm======================
Ylogtis:  (?, 199, 1, 256)
offset:  (256,)
h:  (?, 199, 1, 256)
pooled:  (?, 1, 1, 256)
conv-maxpool-3
filter_shape:  [  3 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-3/W_filter:0' shape=(3, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 198, 1, 256)
======================batchnorm======================
Ylogtis:  (?, 198, 1, 256)
offset:  (256,)
h:  (?, 198, 1, 256)
pooled:  (?, 1, 1, 256)
conv-maxpool-4
filter_shape:  [  4 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-4/W_filter:0' shape=(4, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 197, 1, 256)
======================batchnorm======================
Ylogtis:  (?, 197, 1, 256)
offset:  (256,)
h:  (?, 197, 1, 256)
pooled:  (?, 1, 1, 256)
conv-maxpool-5
filter_shape:  [  5 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-5/W_filter:0' shape=(5, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 196, 1, 256)
======================batchnorm======================
Ylogtis:  (?, 196, 1, 256)
offset:  (256,)
h:  (?, 196, 1, 256)
pooled:  (?, 1, 1, 256)
conv-maxpool-7
filter_shape:  [  7 256   1 256]
W_filter shape:  <tf.Variable 'TextCNN/conv-maxpool-7/W_filter:0' shape=(7, 256, 1, 256) dtype=float32_ref>
beta:  (256,)
conv:  (?, 194, 1, 256)
======================batchnorm======================
Ylogtis:  (?, 194, 1, 256)
offset:  (256,)
h:  (?, 194, 1, 256)
pooled:  (?, 1, 1, 256)
pooled_outputs:  (5,)
h_pool.shape (?, 1, 1, 1280)
h_pool_flat.shape (?, 1280)
output:  (?, 1280)
===============fc-bn-layer===============
W_fc.shape:  (1280, 1024)
h_fc (?, 1024)
beta_fc (1024,)
======================batchnorm======================
Ylogtis:  (?, 1024)
offset:  (1024,)
===============out_layer===============
"""