import os

import tensorflow as tf

from data_loaders.data_generator import DataGenerator
from models.double_lstm_model import DoubleLSTMModel
from trainer.double_lstm_trainer import DoubleLSTMTrainer
from utils.config_util import get_or_create_config, get_config
from utils.utils import clean, create_dirs, mkdir
from utils.vocab_util import get_or_create_vocab, get_vocab

flags = tf.app.flags
flags.DEFINE_boolean("clean", True, "Whether clean train folder")
flags.DEFINE_boolean("train", True, "Whether train the model")

# configurations for the model
flags.DEFINE_integer("embedding_dim", 64, "词向量维度")
flags.DEFINE_integer("num_classes", 2, "类别数")
flags.DEFINE_integer("vocab_size", 10000, "词汇表大小")
flags.DEFINE_integer("num_layers", 2, "隐藏层层数")
flags.DEFINE_integer("hidden_dim", 128, "全连接层神经元")
flags.DEFINE_string("rnn", "lstm", "RNN Cell 类型")

# configurations for training
flags.DEFINE_float("keep_prob", 0.5, "dropout保留比例")
flags.DEFINE_float("learning_rate", 0.001, "学习率")
flags.DEFINE_float("batch_size", 1, "每批训练大小")
flags.DEFINE_integer("num_epochs", 20, "总迭代轮次")
flags.DEFINE_integer("print_per_batch", 100, "每多少轮输出一次结果")
flags.DEFINE_integer("save_per_batch", 10, "每多少轮存入tensorboard")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_integer("max_to_keep", 2, "max_to_keep")

flags.DEFINE_string("tensorboard_dir", os.path.join("tensorboard", "double_lstm"), "TensorBoard Direction")
flags.DEFINE_string("config_file", os.path.join("configs", "double_lstm_config_file"), "模型配置文件")
flags.DEFINE_string("train_file", "data/train_start_boundary.json", "训练集路径")
flags.DEFINE_string("val_file", "data/val_start_boundary.json", "验证集路径")
flags.DEFINE_string("test_file", "data/sample_start_boundary.json", "测试集路径")
flags.DEFINE_string("vocab_dir", "resources", "词汇表路径")
flags.DEFINE_string("checkpoint_dir", os.path.join("checkpoints", "double_lstm"), "最佳验证结果保存路径")

FLAGS = tf.app.flags.FLAGS
assert 0 <= FLAGS.keep_prob < 1, "dropout rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than 0"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
assert FLAGS.rnn in ["gru", "lstm"]


def main_train():
    word_to_id, id_to_word, label_to_id, id_to_label = get_or_create_vocab(FLAGS)
    config = get_or_create_config(FLAGS, id_to_word)

    # create the experiments dirs
    create_dirs([config.tensorboard_dir, config.vocab_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = DoubleLSTMModel(config)
    # load model if exists
    model.load(sess)
    # create your data generator
    train_data = DataGenerator(config, word_to_id, label_to_id, config.train_file)
    val_data = DataGenerator(config, word_to_id, label_to_id, config.val_file)
    # create tensorboard logger
    # logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = DoubleLSTMTrainer(sess, model, train_data, val_data, config, id_to_label)

    # here you train your model
    trainer.train()

    sess.close()


def evaluate_one():
    word_to_id, id_to_word, label_to_id, id_to_label = get_vocab(FLAGS.vocab_dir)
    config = get_config(FLAGS.config_file)

    sess = tf.Session()
    sess.as_default()

    model = DoubleLSTMModel(config)
    model.load(sess)

    while True:
        try:
            left_line = input("input left context: ")
            if left_line == "exit":
                sess.close()
                exit(0)
            right_line = input("input right context: ")
            if right_line == "exit":
                sess.close()
                exit(0)

            if left_line == "":
                left_input = [[word_to_id[left_line]]]
            else:
                left_input = [[word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in left_line]]
            if right_line == "":
                right_input = [[word_to_id[right_line]]]
            else:
                right_input = [[word_to_id[x] if x in word_to_id else word_to_id["<UNK>"] for x in right_line]]

            feed_dict = {
                model.left_x: left_input,
                model.right_x: right_input,
                model.keep_prob: 1.0
            }
            print(left_input)
            print(right_input)
            y_pred_cls, logits = sess.run([model.y_pred_cls, model.logits], feed_dict=feed_dict)
            print(y_pred_cls[0], tf.nn.softmax(logits).eval(session=sess))
            print("所属类别: {}".format(id_to_label[y_pred_cls[0]]))
        except Exception as e:
            sess.close()
            print(e)


def save_for_java():
    config = get_config(FLAGS.config_file)
    sess = tf.Session()
    model = DoubleLSTMModel(config)
    model.load(sess)
    builder = tf.saved_model.builder.SavedModelBuilder("tmp/double_lstm_model")
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING]
    )
    builder.save()
    sess.close()


if __name__ == '__main__':
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
            mkdir()
        main_train()
    else:
        evaluate_one()

        # save_for_java()
