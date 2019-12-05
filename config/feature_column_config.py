#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: feature_column_config.py
@time: 2019/11/11
"""

from itertools import chain

import numpy as np
from tensorflow import feature_column as fc

from config.deep_feature_config import *
from util.cate_values_util import read_feature_values_from_file as read_feature_values

SPARSE_EMBEDDING_SIZE = 8
BUCKTE_EMBEDDING_SIZE = 8

# 配置连续变量的归一化函数和分桶边界
log_features = []
normalizer_features = []
none_features = [x for x in CONTINUOUS_FEATURES if x not in log_features + normalizer_features]

dense_process_dict = dict(
    [(x, lambda x: tf.math.log1p(tf.cast(x, tf.float32))) for x in log_features] +
    [(x, None) for x in normalizer_features] +
    [(x, None) for x in none_features])

dense_bound_dict = {
    "opp_create_obs_interval_minutes": [0.0001] + list(np.arange(1, 8)) + list(np.arange(80, 120, 5) / 10),
    "call_record_tot_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 10)),
    "call_record_eff_num": [0.0001, 3, 6, 10, 20],
    "free_learn_video_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 12)),
    "free_learn_live_length": [0.0001] + list(np.arange(5, 60, 5) / 10) + list(np.arange(6, 12)),
    "tot_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "con_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "stu_chat_count": [0.0001] + list(np.arange(5, 30, 5) / 10) + list(np.arange(3, 7)),
    "stu_con_ratio": [0.0001] + list(np.arange(1, 10) / 100) + list(np.arange(1, 6) / 10)
}

# ===========================离散特征===============================

# sparse feature
sparse_features = \
    [fc.categorical_column_with_vocabulary_list(f_name, read_feature_values(f_name)) for f_name in CATEGORY_FEATURES]

# sparse feature one_hot
sparse_features_one_hot = \
    [fc.indicator_column(feature) for feature in sparse_features]

# sparse feature embedding
sparse_features_emb = \
    [fc.embedding_column(feature, SPARSE_EMBEDDING_SIZE) for feature in sparse_features]

# =============================连续特征=======================================

# dense feature
dense_features = [fc.numeric_column(feature, normalizer_fn=dense_process_dict[feature]) for feature in
                  CONTINUOUS_FEATURES]

# dense embedding
dense_features_emb = [fc.embedding_column(fc.bucketized_column(feature, dense_bound_dict[feature]),
                                          BUCKTE_EMBEDDING_SIZE) for feature in dense_features]

# ========================= LR feature column =============================

LinnerColumns = [
    # sparse ont hot
    sparse_features_one_hot,
    # dense log
    dense_features

]
LinnerColumns = list(chain(*LinnerColumns))

# ========================= FM feature column =============================
InteractionColumns = [
    # sparse emb
    sparse_features_emb,
    # dense emb
    dense_features_emb

]
InteractionColumns_pool = InteractionColumns
InteractionColumns = list(chain(*InteractionColumns))

# ========================= DNN feature column =============================
DNNColumns = InteractionColumns

# 注意池化方法
# print(LinnerColumns)
# print(len(LinnerColumns))
# print(InteractionColumns)
# print(len(InteractionColumns))
