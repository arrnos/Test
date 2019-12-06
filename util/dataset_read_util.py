#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: dataset_read_util.py
@time: 2019/12/06
"""
from config.deep_feature_config import *


# 从csv文件读取dataset
def read_csv_2_dataset(csv_file, mean_dict_save_path, shuffle_size=None, batch_size=256):
    # 加载csv默认值list
    mean_dict = load_mean_dict(mean_dict_save_path)
    csv_feature_defaults = [
        mean_dict[f] if f in mean_dict else FEATURE_DEFAULT_DICT[f] for f in FEATURE_NAME_USE_LIST]
    # print(csv_feature_defaults)

    data = tf.data.TextLineDataset(csv_file)
    if shuffle_size:
        data = data.shuffle(shuffle_size)
    data = data.map(lambda x: _parse_line(x, csv_feature_defaults)).batch(batch_size)
    return data

# 单行解析
def _parse_line(line, csv_feature_defaults):
    # csv_features_defaults和select_cols两个list的长度要保持一致，select_cols存储的是要保留的col的index.
    fields = tf.io.decode_csv(line, csv_feature_defaults, select_cols=FEATURE_USE_LIST)
    features = dict(zip(FEATURE_NAME_USE_LIST, fields))
    label = features.pop("label")
    return features, label

# 加载连续特征均值字典，用于填补缺失值
def load_mean_dict(save_path):
    import os
    import pickle
    assert os.path.exists(save_path)
    save_dict = pickle.load(open(save_path, "rb"))
    mean_dict = {}
    for key, (min_value, max_value, mean_value, _, _) in save_dict.items():
        mean_dict[key] = mean_value
    return mean_dict


if __name__ == '__main__':

    from config.file_path_config import *

    dataset = read_csv_2_dataset(train_csv_file, min_max_value_path, shuffle_size=10000, batch_size=256)
    cnt = 0
    for d in dataset:
        cnt += 1
        print("label")
        print(d[1])
        for k, v in d[0].items():
            print(k)
            print(v)
        if cnt > 1:
            break
