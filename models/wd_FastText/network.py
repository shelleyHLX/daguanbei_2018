# coding: utf-8

import tensorflow as tf
import numpy as np

class Settings(object):
    def __init__(self):
        # self.model_name = 'wd_BiGRU'
        self.model_name = 'wd_FastText_QS200'
        self.fact_len = 800
        # cnn
        self.filter_sizes = [2, 3, 5, 7]  # 尺度
        self.n_filter = 128  # 卷积核的个数

        # FC
        self.hidden_size = 256
        self.n_layer = 1
        self.fc_hidden_size = 1024
        self.n_class = 19
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class FastText(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.n_class = settings.n_class
        self.fact_len = settings.fact_len
        # cnn
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)  # 256 * [2, 3, 4, 5 ,7]
        # fc
        self.fc_hidden_size = 1024
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.l2_loss = tf.constant(0.0)
        # self.l2_reg_lambda = tf.constant(0.4)
        self.update_emas = list()
        self._batch_size = tf.placeholder(tf.int32, [])
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int64, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]
        print('embedding_size: ', self.embedding_size)  # 256
        # 1. embedding layer
        self.embedding_vec = tf.nn.embedding_lookup(self.embedding, self._X_inputs)
        print('embedding_vec: ', self.embedding_vec.shape)  # (?, 200, 256)
        # 2. skip gram
        with tf.variable_scope('skip-gram'):
            print('============skip-gram============')
            self.sentence_embed = tf.reduce_mean(self.embedding_vec, axis=1)
            print('sentence_embed: ',self.sentence_embed.shape)  # (?, 256)
            # dropout
            self.sentence_embed = tf.nn.dropout(self.sentence_embed, self.keep_prob)
            print('dropout: ', self.sentence_embed.shape)  # (?, 256)
            # cnn
        with tf.name_scope('CNN'):
            output = self.CNN(self.sentence_embed)
            # FC

        with tf.variable_scope('fc-bn-layer'):
            print('===============fc-bn-layer===============')
            W_fc = self.weight_variable([512, self.fc_hidden_size], name='Weight_fc')
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

        with tf.variable_scope('output'):
            self.W = self.weight_variable([self.fc_hidden_size, self.n_class], name='W')
            print('W: ', self.W.shape)  # (256, 183)
            tf.summary.histogram('W', self.W)
            self.b = self.bias_variable([self.n_class], name='b')
            print('b: ', self.b.shape)  # (183,)
            tf.summary.histogram('b', self.b)
            self._y_pred = tf.matmul(fc_bn_drop, self.W) + self.b

        with tf.name_scope('loss'):
            self.l2_loss += tf.nn.l2_loss(self.W)
            self.l2_loss += tf.nn.l2_loss(self.b)
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self._y_pred,
                                                        labels=self._y_inputs)
            )
            print('loss: ', self._loss.shape)
            tf.summary.scalar('loss', self._loss)
        self.saver = tf.train.Saver(max_to_keep=1)

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

    @property
    def global_step(self):
        return self._global_step

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def batchnorm(self, Ylogits, offset, convolutional=False):
        """batchnormalization.
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

    def CNN(self, inputs, n_step=16):
        outputs = list()
        inputs = tf.reshape(inputs, shape=[-1, 16, 16])
        inputs = tf.expand_dims(inputs, -1)  # (?, 256, 1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)  # 2，3,4,5,7
                # Convolution Layer
                filter_shape = [filter_size, n_step, 1, self.n_filter]
                print('filter_shape: ', np.asarray(filter_shape))
                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                print('W_filter shape: ', W_filter)
                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')
                print('beta: ', beta.shape)  # (256,)(256,)(256,)
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                print('conv: ', conv.shape)
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply nonlinearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                print('h: ', h.shape)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                print('pooled: ', pooled.shape)
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        print('pooled_outputs: ', np.asarray(pooled_outputs).shape) #  (5,)
        h_pool = tf.concat(pooled_outputs, 3)
        print('h_pool.shape', h_pool.shape)  # (?, 1, 1, 1280) 1280 = 256 × 5
        h_pool_flat = tf.reshape(h_pool, [-1, 512])
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]


def FastText_test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    W_embedding = np.random.randn(50, 256)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = FastText(W_embedding, settings)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(model.loss)
        update_op = tf.group(*model.update_emas)
        sess.run(tf.global_variables_initializer())
        fetch = [model.loss, model.y_pred, train_op, update_op]
        loss_list = list()
        for i in range(10000):
            X_batch = np.zeros((batch_size, 200), dtype=float)
            y_batch = np.zeros((batch_size, 183), dtype=int)
            _batch_size = len(y_batch)
            feed_dict = {model.X_inputs: X_batch, model.y_inputs: y_batch,
                         model.batch_size: _batch_size, model.tst: False, model.keep_prob: 0.5}
            loss, y_pred, _, _ = sess.run(fetch, feed_dict=feed_dict)
            loss_list.append(loss)
            print(i, loss)

if __name__ == '__main__':
    FastText_test()

