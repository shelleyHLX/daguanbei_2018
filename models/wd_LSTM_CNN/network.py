# -*- coding:utf-8 -*-

import tensorflow as tf

"""LSTM_CNN
"""

# 输入shape = (batch_size, time_step, input_size)  (batch_size,行数,每一行的维度)

class Settings(object):
    def __init__(self):
        self.model_name = 'wd_LSTM_CNN_pdQS200'
        # LSTM
        self.time_step = 200  # 单词个数
        self.input_size = 256  # 数据维度
        self.lstm_hidden_neural_size = 256  # 256
        self.lstm_hidden_layer_num = 2  # 2
        # CNN
        self.hidden_size = 256
        self.n_layer = 1
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        # FC
        self.n_class = 202
        self.fc_hidden_layer_num = 1
        self.fc_hidden_neural_size = 1024
        # SAVE
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class LSTM_CNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        # lstm
        self.time_step = settings.time_step
        self.input_size = settings.input_size
        self.lstm_hidden_neural_size = settings.lstm_hidden_neural_size  # LSTM的隐藏层个数
        self.lstm_hidden_layer_num = settings.lstm_hidden_layer_num  # LSTM layer 的层数
        # CNN
        self.hidden_size = settings.hidden_size
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)  # 256 * 5 = 1280
        # fc
        self.fc_hidden_neural_size = settings.fc_hidden_neural_size

        self.n_class = settings.n_class
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()

        # placeholders
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int64, [None, self.time_step], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                                   initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]
        print('W_embedding.shape', W_embedding.shape)

        with tf.variable_scope('LSTM'):
            print('===============LSTM===============')
            output_lstm = self.LSTM(self._X_inputs)
            print('output_lstm: ', output_lstm.shape)

        with tf.variable_scope('TextCNN'):
            print('============== TextCNN ==============')
            output_cnn = self.TextCNN(self._X_inputs, self.time_step)
            print('output_cnn', output_cnn.shape)

        with tf.variable_scope('fc-bn-layer'):
            output = tf.concat([output_lstm, output_cnn], axis=1)
            print('[output_lstm, output_cnn].shape', output.shape)
            print('===============fc-bn-layer===============')
            W_fc = self.weight_variable([self.lstm_hidden_neural_size + self.n_filter_total, self.fc_hidden_neural_size],
                                        name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_neural_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")
            fc_bn_drop = tf.nn.dropout(self.fc_bn_relu, self.keep_prob)

        with tf.variable_scope('out_layer'):
            print('===============out_layer===============')
            W_out = self.weight_variable([self.fc_hidden_neural_size, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(fc_bn_drop, W_out, b_out, name='y_pred')  # 每个类别的分数 scores

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
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           self._global_step)  # adding the iteration prevents from averaging across non-existing iterations
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

    def lstm_cell(self, lstm_hidden_neural_size):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        return drop

    def LSTM(self, X_inputs):
        # build LSTM network
        print('========================LSTM========================')
        print('X_inputs.shape', X_inputs.shape)
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        m_lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [self.lstm_cell(self.lstm_hidden_neural_size)
             for _ in range(self.lstm_hidden_layer_num)],
                                           state_is_tuple=True)
        self._initial_state = m_lstm_cell.zero_state(self.batch_size, tf.float32)
        out_put = []
        state = self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step_ in range(self.time_step):
                if time_step_ > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = m_lstm_cell(inputs[:, time_step_, :], state)
                out_put.append(cell_output)
        output = out_put [-1]  # (?, lstm_hidden_neural_size)
        return output

    def TextCNN(self, cnn_inputs, n_step):
        print('===================textcnn===================')
        """build the TextCNN network. Return the h_drop"""
        # cnn_inputs.shape = [batchsize, n_step, hidden_size*2+embedding_size]
        # # (?, 200, 768)
        inputs = tf.nn.embedding_lookup(self.embedding, cnn_inputs)
        print('inputs.shape: ', inputs.shape)
        inputs = tf.expand_dims(inputs, -1)
        print('inputs.shape: ', inputs.shape)
        # print('inputs.shape: ', inputs.shape)  # (?, 200, 256)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)  # 2，3,4,5,7
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


# test the model
def LSTM_CNN_test():
    import numpy as np
    print('Begin testing...')
    settings = Settings()
    W_embedding = np.random.randn(50, 256)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_size = 128
    with tf.Session(config=config) as sess:
        model = LSTM_CNN(W_embedding, settings)
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
    LSTM_CNN_test()
