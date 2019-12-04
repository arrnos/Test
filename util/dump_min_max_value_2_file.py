#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: dump_category_value_2_file.py
@time: 2019/12/02
"""

from config.deep_feature_config import *
from config.file_path_config import category_value_path
import os
from util.load_dict import load_dict
from codecs import open
from collections import defaultdict
import numpy as np
import tensorflow as tf
import pickle


def dump_min_max_values_2_file(csv_file, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rs_dict = {}
    for f in CONTINUOUS_FEATURES:
        rs_dict[f] = [np.inf, -np.inf]

    with open(csv_file, "r", "utf-8") as fin:
        for i, line in enumerate(fin):
            arr = line.strip().split(",")
            features = dict(zip(FEATURE_NAMES, arr))
            for f in CONTINUOUS_FEATURES:
                # 每种特征，统计其每种特征值的出现的最小值和最大值
                f_value = features[f]
                if not f_value:
                    continue
                if FEATURE_DTYPE_DICT[f] == tf.int64:
                    f_value = int(f_value)
                if FEATURE_DTYPE_DICT[f] == tf.float32:
                    f_value = float(f_value)
                rs_dict[f][0] = min(rs_dict[f][0], f_value)
                rs_dict[f][1] = max(rs_dict[f][1], f_value)

            if i > 0 and i % 5000 == 0:
                print("line:", i)

    pickle.dump(rs_dict, open(out_path, "wb"))

    print("min max value dump completed!", out_path)


def load_min_max_value_dict_from_file(min_max_value_save_path=min_max_value_path):
    assert os.path.exists(min_max_value_save_path)
    return pickle.load(open(min_max_value_path, "rb"))


if __name__ == '__main__':
    from config.file_path_config import *

    dump_min_max_values_2_file(train_csv_file, min_max_value_path)

    rs_dict = load_min_max_value_dict_from_file(min_max_value_path)
    print(rs_dict)
    print(rs_dict["account_age"])
