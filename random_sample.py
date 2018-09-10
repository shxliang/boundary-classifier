import codecs
import json

import numpy as np


def main():
    with codecs.open("data/0-620_start_boundary.json", mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    train_set = []
    val_set = []

    for line in lines:
        cur_json = json.loads(line)
        cur_label = cur_json["label"]
        if cur_label == "1":
            prob = np.random.random()
            if prob <= 0.7:
                train_set.append(line)
            else:
                val_set.append(line)
        else:
            prob = np.random.random()
            if prob <= 0.06:
                prob = np.random.random()
                if prob <= 0.7:
                    train_set.append(line)
                else:
                    val_set.append(line)

    with codecs.open("data/train_start_boundary.json", mode="w", encoding="utf-8") as f:
        f.writelines(train_set)
    with codecs.open("data/val_start_boundary.json", mode="w", encoding="utf-8") as f:
        f.writelines(val_set)


if __name__ == "__main__":
    main()
