from tqdm import tqdm

from bases.base_trainer import BaseTrain
import numpy as np
from sklearn import metrics


class DoubleLSTMTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, val_data, config, id_to_label):
        super(DoubleLSTMTrainer, self).__init__(sess, model, train_data, config)
        self.val_data = val_data
        self.id_to_label = id_to_label

    def train_epoch(self):
        train_batch_iter = self.data.iter_batch(shuffle=True)

        loop = tqdm(range(self.data.num_batch))
        losses = []
        accs = []
        pred_ys = []
        truth_ys = []

        for _ in loop:
            self.train_batch = next(train_batch_iter)
            loss, acc = self.train_step()
            pred_y = self.evaluate_step(self.train_batch)[0]

            losses.append(loss)
            accs.append(acc)
            pred_ys.extend(pred_y)
            truth_ys.extend(np.argmax(self.train_batch[2], axis=1))

        loss = np.mean(losses)
        acc = np.mean(accs)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess) + 1

        print(metrics.classification_report(truth_ys, pred_ys, target_names=self.id_to_label))
        print("epoch: {}/{}, loss: {}, acc: {}".format(cur_epoch, self.config.num_epochs, loss, acc))

        val_truth_ys = np.argmax(self.val_data.batch_data[2], axis=1)
        val_pred_ys = self.evaluate_all(self.val_data)
        print(metrics.classification_report(val_truth_ys, val_pred_ys, target_names=self.id_to_label))

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_left = self.train_batch[0]
        batch_right = self.train_batch[1]
        batch_label = self.train_batch[2]

        feed_dict = {self.model.left_x: batch_left, self.model.right_x: batch_right, self.model.y: batch_label,
                     self.model.keep_prob: self.config.keep_prob, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.optim, self.model.loss, self.model.acc], feed_dict=feed_dict)
        return loss, acc

    def evaluate_step(self, batch):
        batch_left = batch[0]
        batch_right = batch[1]
        batch_label = batch[2]

        feed_dict = {self.model.left_x: batch_left, self.model.right_x: batch_right, self.model.y: batch_label,
                     self.model.keep_prob: 1.0, self.model.is_training: False}
        pred_y = self.sess.run([self.model.y_pred_cls], feed_dict=feed_dict)
        return pred_y

    def evaluate_all(self, data):
        all_left = data[0]
        all_right = data[1]
        all_label = data[2]

        pred_ys = []

        for left, right, label in all_left, all_right, all_label:
            feed_dict = {self.model.left_x: [left], self.model.right_x: [right], self.model.y: [label],
                         self.model.keep_prob: 1.0, self.model.is_training: False}
            pred_y = self.sess.run([self.model.y_pred_cls], feed_dict=feed_dict)
            pred_ys.append(pred_y)
        return pred_ys
