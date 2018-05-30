import codecs
import json
from collections import Counter
import numpy as np


def main():
    with codecs.open("data/0-620_start_boundary.json", mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    result = []
    for line in lines:
        cur_json = json.loads(line)
        cur_label = cur_json["label"]
        if cur_label == "1":
            result.append(line)
        else:
            prob = np.random.random()
            if prob <= 0.06:
                result.append(line)

    with codecs.open("data/sample_start_boundary.json", mode="w", encoding="utf-8") as f:
        f.writelines(result)


if __name__ == "__main__":
    main()
