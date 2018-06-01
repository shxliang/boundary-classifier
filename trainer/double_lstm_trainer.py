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
        train_losses = []
        train_accs = []
        train_pred_ys = []
        train_truth_ys = []

        for _ in loop:
            self.train_batch = next(train_batch_iter)
            train_loss, train_acc = self.train_step()
            pred_y = self.evaluate_step(self.train_batch)[0]

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_pred_ys.extend(pred_y)
            train_truth_ys.extend(np.argmax(self.train_batch[2], axis=1))

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess) + 1

        val_truth_ys = np.argmax([l[2][0] for l in self.val_data.batch_data], axis=1)
        val_pred_ys = self.evaluate_all(self.val_data)
        val_acc = metrics.accuracy_score(val_truth_ys, val_pred_ys)

        print("train PRF report:")
        print(metrics.classification_report(train_truth_ys, train_pred_ys, target_names=self.id_to_label))
        print("val PRF report:")
        print(metrics.classification_report(val_truth_ys, val_pred_ys, target_names=self.id_to_label))
        print("epoch: {}/{}, train loss: {}, train acc: {}, val acc: {}".format(cur_epoch, self.config.num_epochs,
                                                                                train_loss, train_acc, val_acc))

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': train_loss,
            'acc': train_acc,
        }
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        self.model.save(self.sess)

    def train_step(self):
        batch_left = self.train_batch[0]
        batch_right = self.train_batch[1]
        batch_label = self.train_batch[2]

        feed_dict = {self.model.left_x: batch_left, self.model.right_x: batch_right, self.model.y: batch_label,
                     self.model.keep_prob: self.config.keep_prob}
        _, loss, acc = self.sess.run([self.model.optim, self.model.loss, self.model.acc], feed_dict=feed_dict)
        return loss, acc

    def evaluate_step(self, batch):
        batch_left = batch[0]
        batch_right = batch[1]
        batch_label = batch[2]

        feed_dict = {self.model.left_x: batch_left, self.model.right_x: batch_right, self.model.y: batch_label,
                     self.model.keep_prob: 1.0}
        pred_y = self.sess.run([self.model.y_pred_cls], feed_dict=feed_dict)
        return pred_y

    def evaluate_all(self, data):
        pred_ys = []
        for cur_data in data.batch_data:
            cur_left = cur_data[0]
            cur_right = cur_data[1]
            cur_label = cur_data[2]

            feed_dict = {self.model.left_x: cur_left, self.model.right_x: cur_right, self.model.y: cur_label,
                         self.model.keep_prob: 1.0}
            pred_y = self.sess.run([self.model.y_pred_cls], feed_dict=feed_dict)[0]
            pred_ys.extend(pred_y)
        return pred_ys
