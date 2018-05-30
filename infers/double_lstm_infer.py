import os
import tensorflow as tf

from bases.base_infer import InferBase


class DoubleLSTMInfer(InferBase):
    def __init__(self, sess, model, config):
        super(DoubleLSTMInfer, self).__init__(config)
        self.sess = sess
        self.model = model
        self.load_model(sess, config.checkpoint_dir)

    def load_model(self, sess, model_path):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=model_path)

    def predict(self, data):
        pass


    def peridct_one(self, data):
        data = [word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in line]
        pad_data = kr.preprocessing.sequence.pad_sequences([data], config.seq_length, padding="post",
                                                           truncating="post")
        print(pad_data)
        feed_dict = {
            model.input_x: pad_data,
            model.keep_prob: 1.0
        }
        y_pred_cls, logits = session.run([model.y_pred_cls, model.logits], feed_dict=feed_dict)

