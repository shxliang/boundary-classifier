import codecs
import os
import json
from collections import OrderedDict

from bunch import Bunch


def get_or_create_config(params, id_to_word):
    if os.path.exists(params.config_file):
        config = get_config(params.config_file)
    else:
        config = create_config(params, id_to_word)
    return config


def get_config(config_file: str):
    with codecs.open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = Bunch(config_dict)
    return config


def create_config(params, id_to_word):
    config_dict = OrderedDict()

    config_dict["embedding_dim"] = params.embedding_dim
    config_dict["num_classes"] = params.num_classes
    config_dict["vocab_size"] = len(id_to_word)
    config_dict["num_layers"] = params.num_layers
    config_dict["hidden_dim"] = params.hidden_dim
    config_dict["rnn"] = params.rnn

    config_dict["learning_rate"] = params.learning_rate
    config_dict["keep_prob"] = params.keep_prob
    config_dict["batch_size"] = params.batch_size
    config_dict["num_epochs"] = params.num_epochs
    config_dict["print_per_batch"] = params.print_per_batch
    config_dict["save_per_batch"] = params.save_per_batch
    config_dict["optimizer"] = params.optimizer
    config_dict["max_to_keep"] = params.max_to_keep

    config_dict["tensorboard_dir"] = params.tensorboard_dir
    config_dict["train_file"] = params.train_file
    config_dict["val_file"] = params.val_file
    config_dict["test_file"] = params.test_file
    config_dict["checkpoint_dir"] = params.checkpoint_dir
    config_dict["vocab_dir"] = params.vocab_dir

    save_config(config_dict, params.config_file)
    config = Bunch(config_dict)
    return config


def save_config(config_dict, config_file):
    with codecs.open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
