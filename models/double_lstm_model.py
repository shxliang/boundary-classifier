from bases.base_model import BaseModel
import tensorflow as tf


class DoubleLSTMModel(BaseModel):
    def __init__(self, config):
        super(DoubleLSTMModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        def lstm_cell():  # lstm cell
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru cell
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 创建rnn cell，并为每一个rnn cell后面加一个dropout层
            if self.config.rnn == "lstm":
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        self.left_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name="left_x")
        self.right_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name="right_x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.config.num_classes], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.is_training = tf.placeholder(dtype=tf.bool)

        # 词向量embedding layer
        with tf.device('/cpu:0'):
            left_embedding = tf.get_variable("left_embedding", [self.config.vocab_size, self.config.embedding_dim])
            left_embedding_inputs = tf.nn.embedding_lookup(left_embedding, self.left_x)

            right_embedding = tf.get_variable("right_embedding", [self.config.vocab_size, self.config.embedding_dim])
            right_embedding_inputs = tf.nn.embedding_lookup(right_embedding, self.left_x)

        with tf.name_scope("rnn"):
            with tf.variable_scope("left_rnn"):
                left_cells = [dropout() for _ in range(self.config.num_layers)]
                left_rnn_cell = tf.contrib.rnn.MultiRNNCell(left_cells, state_is_tuple=True)
                # LSTM时final_state是个tuple(c,h)，c和h的维度都是hidden_dim
                # GRU时final_state维度就是hidden_dim
                left_outputs, left_final_state = tf.nn.dynamic_rnn(cell=left_rnn_cell, inputs=left_embedding_inputs,
                                                                   dtype=tf.float32)
                left_last = left_outputs[:, -1, :]  # 取最后一个时序输出作为结果

            with tf.variable_scope("right_rnn"):
                right_cells = [dropout() for _ in range(self.config.num_layers)]
                right_rnn_cell = tf.contrib.rnn.MultiRNNCell(right_cells, state_is_tuple=True)
                right_outputs, right_final_state = tf.nn.dynamic_rnn(cell=right_rnn_cell, inputs=right_embedding_inputs,
                                                                     dtype=tf.float32)
                right_last = right_outputs[:, -1, :]

            last = tf.concat([left_last, right_last], axis=-1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 全连接层
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 计算交叉熵损失函数
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 定义优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 计算正确率
            correct_pred = tf.equal(tf.argmax(self.y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
