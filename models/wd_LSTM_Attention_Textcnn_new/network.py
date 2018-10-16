# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

"""
wd_BiGRU_Attention
bigru + attention
"""

class Settings(object):
    def __init__(self):
        # self.model_name = 'wd_BiGRU'
        self.model_name = 'wd_LSTM_Attention_textcnn_new_con_pdSQ200'
        self.fact_len = 200

        # LSTM
        self.time_step = 200  # 单词个数
        self.input_size = 256  # 数据维度
        self.lstm_hidden_neural_size = 256  # 256
        self.lstm_hidden_layer_num = 2  # 2
        self.hidden_size = 512
        # cnn
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.fc_hidden_size = 1024

        # FC
        self.n_layer = 1
        self.fc_hidden_size = 1024
        self.n_class = 183

        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class LSTM_Atten(object):
    """
    fact: inputs -> bigru+attention -> output
    output -> fc+bn+relu -> sigmoid_entropy.
    """
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        # lstm
        self.time_step = settings.time_step
        self.input_size = settings.input_size
        self.lstm_hidden_neural_size = settings.lstm_hidden_neural_size  # LSTM的隐藏层个数
        self.lstm_hidden_layer_num = settings.lstm_hidden_layer_num  # LSTM layer 的层数
        # cnn
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)  # 256 * [2, 3, 4, 5 ,7]

        self.hidden_size = settings.hidden_size
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
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('BiGRU'):
            output = self.bigru_inference(self._X_inputs)
            print('output: ', output.shape)

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

    def lstm_cell(self, lstm_hidden_neural_size):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        return drop

    def LSTM(self, inputs):
        # build LSTM network
        print('======================== LSTM ========================')
        print('X_inputs.shape', inputs.shape)
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
                # print('type(cell_output)', type(cell_output))  # Tensor
                # print('cell_output.shape', cell_output.shape)  # (?, 256)
                out_put.append(cell_output)
        # output = out_put [-1]  # (?, lstm_hidden_neural_size)
        # output = out_put
        out2 = tf.convert_to_tensor(out_put)  # out2:  (200, ?, 256)
        out3 = tf.transpose(out2, perm=[1, 0, 2])  # wonderful
        print('out2: ', out2.shape)
        print('out3: ', out3.shape)
        print('out_put: ', np.asarray(out_put).shape)
        # out1 = tf.reduce_sum(out_put, axis=0)  # 1 (200, 256) 2 (200, ?) 1 (?, 256)
        # print('out1.shape', out1.shape)
        return out3

    def task_specific_attention(self, inputs, output_size,
                                initializer=layers.xavier_initializer(),
                                activation_fn=tf.tanh, scope=None):
        """
        Performs task-specific attention reduction, using learned
        attention context vector (constant within task of interest).
        Args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
           outputs: Tensor of shape [batch_size, output_dim].
        """
        print('============ task_specific_attention ==============')
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        print('inputs: ', inputs.shape)
        print('output_size: ', output_size)
        with tf.variable_scope(scope or 'attention') as scope:
            # u_w, attention 向量
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                       initializer=initializer, dtype=tf.float32)
            print('attention_context_vector', attention_context_vector.shape)
            # 全连接层，把 h_i 转为 u_i ， shape= [batch_size, units, input_size] -> [batch_size, units, output_size]
            input_projection = layers.fully_connected(inputs, output_size,
                                                      activation_fn=activation_fn, scope=scope)
            print('input_projection: ', input_projection.shape)
            # 输出 [batch_size, units]
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector),
                                        axis=2, keep_dims=True)
            print('vector_attn: ', vector_attn.shape)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            print('attention_weights: ', attention_weights.shape)
            tf.summary.histogram('attention_weigths', attention_weights)
            weighted_projection = tf.multiply(inputs, attention_weights)
            # print('weighted_projection: ', weighted_projection.shape)
            # outputs = tf.reduce_sum(weighted_projection, axis=1)
            # print('outputs: ', outputs.shape)
            return weighted_projection  # 输出 [batch_size, self.hidden_size*2]
    def cnn_inference(self, X_inputs, n_step):  # fact_len = n_step
        """TextCNN 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        print('======================into cnn_inference======================')
        # print('inputs.shape: ', inputs.shape)  # (?, 200, 256)
        inputs = tf.expand_dims(X_inputs, -1)
        print('inputs.shape: ', inputs.shape)  # (?, 200, 256, 1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size) # 2，3,4,5,7
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                # print('filter_shape: ', np.asarray(filter_shape))
                # [  2 256   1 256]
                # [  3 256   1 256]
                # [  4 256   1 256]
                # [  5 256   1 256]
                # [  7 256   1 256]
                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                # print('W_filter shape: ', W_filter)
                #  (2, 256, 1, 256)
                # (3, 256, 1, 256)
                # (4, 256, 1, 256)
                # (7, 256, 1, 256)
                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')
                # print('beta: ', beta.shape)  #  (256,)(256,) (256,)
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # print('conv: ', conv.shape)
                #  (?, 199, 1, 256)
                # (?, 198, 1, 256)
                # (?, 197, 1, 256)
                # (?, 196, 1, 256)
                # (?, 196, 1, 256)
                # (?, 194, 1, 256)
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)  # 在激活层前面加 BN
                # Apply non-linearity, batch norm scaling is not useful with relus
                # batch norm offsets are used instead of biases,使用 BN 层的 offset，不要 biases
                h = tf.nn.relu(conv_bn, name="relu")
                # print('h: ', h.shape)
                #  (?, 199, 1, 256)
                # (?, 198, 1, 256)
                # (?, 197, 1, 256)
                # (?, 196, 1, 256)
                # (?, 194, 1, 256)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # print('pooled: ', pooled.shape)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                # (?, 1, 1, 256)
                #  (?, 1, 1, 256)
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        # print('pooled_outputs: ', np.asarray(pooled_outputs).shape) #  (5,)
        h_pool = tf.concat(pooled_outputs, 3)
        # print('h_pool.shape', h_pool.shape)  # (?, 1, 1, 1280) 1280 = 256 × 5
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        # print('h_pool_flat.shape', h_pool_flat.shape)  # (?, 1280)
        return h_pool_flat  # shape = [batch_size, self.n_filter_total]

    def bigru_inference(self, X_inputs):
        print('============ bigru_inference =============')
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        print('X_inputs: ', X_inputs.shape)
        print('inputs: ', inputs.shape)
        output_lstm = self.LSTM(inputs)
        print('output_lstm: ', output_lstm.shape)  # (?, 200, self.hidden_size*2)
        output_att = self.task_specific_attention(inputs, self.hidden_size*2)
        print('output_att: ', output_att.shape) # (?, 200, 256)
        # output = tf.concat([output_lstm, output_att], axis=2)
        out1 = tf.add(output_lstm, output_att)
        out1 = tf.divide(out1, 2)
        print('out1: ', out1.shape)
        # print('output: ', output.shape)
        output_cnn = self.cnn_inference(out1, self.fact_len)
        print('output_cnn: ', output_cnn.shape)
        return output_cnn
"""
============ bigru_inference =============
X_inputs:  (?, 200)
inputs:  (?, 200, 256)
======================== LSTM ========================
X_inputs.shape (?, 200, 256)
out2:  (200, ?, 256)
out3:  (?, 200, 256)
out_put:  (200,)
output_lstm:  (?, 200, 256)
============ task_specific_attention ==============
inputs:  (?, 200, 256)
output_size:  1024
attention_context_vector (1024,)
input_projection:  (?, 200, 1024)
vector_attn:  (?, 200, 1)
attention_weights:  (?, 200, 1)
output_att:  (?, 200, 256)
======================into cnn_inference======================
inputs.shape:  (?, 200, 256, 1)
conv-maxpool-2
================= batch norm ================
conv-maxpool-3
================= batch norm ================
conv-maxpool-4
================= batch norm ================
conv-maxpool-5
================= batch norm ================
conv-maxpool-7
================= batch norm ================
output_cnn:  (?, 1280)
output:  (?, 1280)
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
        model = LSTM_Atten(W_embedding, settings)
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

"""
============ bigru_inference =============
X_inputs:  (?, 200)
inputs:  (?, 200, 256)
======================== LSTM ========================
X_inputs.shape (?, 200, 256)
out2:  (200, ?, 256)
out3:  (?, 200, 256)
out_put:  (200,)
output_lstm:  (?, 200, 256)
============ task_specific_attention ==============
inputs:  (?, 200, 256)
output_size:  1024
attention_context_vector (1024,)
input_projection:  (?, 200, 1024)
vector_attn:  (?, 200, 1)
attention_weights:  (?, 200, 1)
output_att:  (?, 200, 256)
======================into cnn_inference======================
inputs.shape:  (?, 200, 256, 1)
conv-maxpool-2
================= batch norm ================
conv-maxpool-3
================= batch norm ================
conv-maxpool-4
================= batch norm ================
conv-maxpool-5
================= batch norm ================
conv-maxpool-7
================= batch norm ================
output_cnn:  (?, 1280)
output:  (?, 1280)
================= batch norm ================
"""