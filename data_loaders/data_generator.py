import math
import random

from utils.utils import read_json_file, parse_one_hot


class DataGenerator:
    def __init__(self, config, word_to_id: dict, label_to_id: dict, data_path: str):
        self.config = config
        self.word_to_id = word_to_id
        self.label_to_id = label_to_id
        self.data_path = data_path
        self.batch_data = self.sort_and_pad()

    def sort_and_pad(self):
        left, right, label = read_json_file(self.data_path)
        label = parse_one_hot(label, self.label_to_id)
        data = list(zip(left, right, label))

        self.len_data = len(data)
        # 计算一个epoch有几个batch
        self.num_batch = int(math.ceil(len(data) / self.config.batch_size))
        # 以序列长度升序排序
        sorted_data = sorted(data, key=lambda x: len(x[0]) + len(x[1]))
        batch_data = list()
        # 这里只进行了排序，不先进行shuffle，目的是使同一batch中序列长度差别不大
        # 对同一batch里的序列使用0进行padding，padding到当前batch中最长序列长度
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i * self.config.batch_size: (i + 1) * self.config.batch_size],
                                            self.word_to_id, self.label_to_id))
        return batch_data

    @staticmethod
    def pad_data(data, word_to_id: dict, label_to_id: dict):
        padded_left = []
        padded_right = []
        labels = []

        left_max_length = max([len(sentence[0]) for sentence in data])
        right_max_length = max([len(sentence[1]) for sentence in data])

        for line in data:
            left, right, label = line
            left = [word_to_id[w] if w in word_to_id else word_to_id["<UNK>"] for w in [l for l in left]]
            right = [word_to_id[w] if w in word_to_id else word_to_id["<UNK>"] for w in [r for r in right]]

            left_padding = [0] * (left_max_length - len(left))
            right_padding = [0] * (right_max_length - len(right))

            padded_left.append(left + left_padding)
            padded_right.append(right + right_padding)
            labels.append(label)

        return [padded_left, padded_right, labels]

    def iter_batch(self, shuffle=False):
        # 是否对batch顺序进行shuffle
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
