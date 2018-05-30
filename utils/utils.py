import codecs
import json
import os
import shutil

import numpy as np


def read_json_file(file_path):
    """
    读取数据文件，文件为json格式，字段包含left、right、label
    :param file_path: 
    :return: 
    """
    with codecs.open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        jsons = [json.loads(line) for line in lines]
        left = [j["left"] for j in jsons]
        right = [j["right"] for j in jsons]
        label = [j["label"] for j in jsons]
    return left, right, label


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def parse_one_hot(label: list, label_to_id: dict):
    one_hot_list = [[0 for _ in range(len(label_to_id))] for _ in range(len(label))]
    for i in range(len(one_hot_list)):
        one_hot_list[i][label_to_id[label[i]]] = 1
    return one_hot_list


def clean(params):
    if os.path.isdir(params.vocab_dir):
        shutil.rmtree(params.vocab_dir)

    if os.path.isdir(params.checkpoint_dir):
        shutil.rmtree(params.checkpoint_dir)

    if os.path.isdir(params.tensorboard_dir):
        shutil.rmtree(params.tensorboard_dir)

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")
