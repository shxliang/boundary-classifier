import codecs
import itertools
import os
from collections import Counter

from utils.utils import read_json_file


def get_or_create_vocab(params):
    if os.path.exists(os.path.join(params.vocab_dir, "vocab.txt")) and os.path.exists(
            os.path.join(params.vocab_dir, "label.txt")):
        word_to_id, id_to_word, label_to_id, id_to_label = get_vocab(params.vocab_dir)
    else:
        word_to_id, id_to_word, label_to_id, id_to_label = create_vocab(params)
    return word_to_id, id_to_word, label_to_id, id_to_label


def get_vocab(vocab_dir: str):
    vocab_file = os.path.join(vocab_dir, "vocab.txt")
    label_file = os.path.join(vocab_dir, "label.txt")

    if not os.path.exists(vocab_file):
        raise Exception("not exist vocab.txt")
    if not os.path.exists(label_file):
        raise Exception("not exist label.txt")

    with codecs.open(vocab_file, "r", encoding="utf-8") as f:
        id_to_word = f.readlines()
        id_to_word = [w.strip() for w in id_to_word]
    word_to_id = dict(zip(id_to_word, range(len(id_to_word))))

    with codecs.open(label_file, "r", encoding="utf-8") as f:
        id_to_label = f.readlines()
        id_to_label = [w.strip() for w in id_to_label]
    label_to_id = dict(zip(id_to_label, range(len(id_to_label))))

    return word_to_id, id_to_word, label_to_id, id_to_label


def create_vocab(params):
    if not os.path.exists(params.vocab_dir):
        os.mkdir(params.vocab_dir)

    left, right, label = read_json_file(params.train_file)
    left = [[w.strip() for w in l] for l in left]
    left = list(itertools.chain(*left))
    right = [[w.strip() for w in l] for l in right]
    right = list(itertools.chain(*right))

    all_content = []
    all_content.extend(left)
    all_content.extend(right)
    content_counter = Counter(all_content)
    content_count_pairs = content_counter.most_common(params.vocab_size - 1)
    id_to_word, _ = list(zip(*content_count_pairs))
    id_to_word = ["<PAD>", "<UNK>"] + list(id_to_word)
    save_vocab(id_to_word, os.path.join(params.vocab_dir, "vocab.txt"))
    word_to_id = dict(zip(id_to_word, range(len(id_to_word))))

    label_counter = Counter(label)
    label_count_pairs = label_counter.most_common(params.num_classes)
    id_to_label, _ = list(zip(*label_count_pairs))
    save_vocab(id_to_label, os.path.join(params.vocab_dir, "label.txt"))
    label_to_id = dict(zip(id_to_label, range(len(id_to_label))))

    return word_to_id, id_to_word, label_to_id, id_to_label


def save_vocab(vocab, file_path):
    with codecs.open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
