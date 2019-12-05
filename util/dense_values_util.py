#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: cate_values_util.py
@time: 2019/12/02
"""

import pickle
from codecs import open

import numpy as np

from config.deep_feature_config import *
from config.file_path_config import *


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
                f_value = float(f_value)
                rs_dict[f][0] = min(rs_dict[f][0], f_value)
                rs_dict[f][1] = max(rs_dict[f][1], f_value)

            if i > 0 and i % 5000 == 0:
                print("line:", i)

    pickle.dump(rs_dict, open(out_path, "wb"))

    print("min max value dump completed!", out_path)


def load_min_max_value_dict_from_file(min_max_value_save_path):
    assert os.path.exists(min_max_value_save_path)
    return pickle.load(open(min_max_value_save_path, "rb"))


def prepare_log_min_max_dict(min_max_dict):
    log_config_dict = {}
    for feature, (min_value, max_value) in min_max_dict.items():
        log_config_dict[feature] = (np.log1p(min_value), np.log1p(max_value))
    return log_config_dict


min_max_dict = load_min_max_value_dict_from_file(min_max_value_path)
log_min_max_dict = prepare_log_min_max_dict(min_max_dict)


def min_max_norm_func(f):
    min_value, max_value = min_max_dict[f]
    return lambda x: 1.0 * (x - min_value) / (max_value - min_value)


def log_min_max_func(f):
    min_value, max_value = log_min_max_dict[f]
    return lambda x: 1.0 * (np.log1p(x) - min_value) / (max_value - min_value)


dense_process_dict = dict(
    [(f, log_min_max_func(f)) for f in LOG_MIN_MAX_METHOD_LIST] +
    [(f, min_max_norm_func(f)) for f in MIN_MAX_METHOD_LIST])

if __name__ == '__main__':
    dump_min_max_values_2_file(train_csv_file, min_max_value_path)

    print(min_max_dict)
    print(log_min_max_dict)
    print(min_max_dict["account_age"])
    print(log_min_max_dict["account_age"])
