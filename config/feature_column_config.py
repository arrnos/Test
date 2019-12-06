#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: feature_column_config.py
@time: 2019/11/11
"""

from itertools import chain

from tensorflow import feature_column as fc

from config.deep_feature_config import *
from util.cate_values_util import read_feature_values_from_file as read_feature_values
from util.dense_values_util import dense_process_dict

SPARSE_EMBEDDING_SIZE = 8
BUCKTE_EMBEDDING_SIZE = 8

# 配置连续变量的归一化函数和分桶边界
dense_process_dict = dense_process_dict
dense_bound_list = [x / 10 for x in range(1, 10)]

# =============================连续特征=======================================

# dense feature
dense_features = [fc.numeric_column(feature, normalizer_fn=dense_process_dict[feature]) for feature in
                  CONTINUOUS_FEATURES]

# dense embedding
dense_features_emb = [fc.embedding_column(fc.bucketized_column(feature, dense_bound_list),
                                          BUCKTE_EMBEDDING_SIZE) for feature in dense_features]

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

# ========================= LR feature column =============================

LinerColumns = [
    # sparse_features_one_hot,
    dense_features
]

LinerColumns = list(chain(*LinerColumns))

# ========================= FM feature column =============================
InteractionColumns = [
    sparse_features_emb,
    # dense_features_emb
]

InteractionColumns = list(chain(*InteractionColumns))

# ========================= DNN feature column =============================
DNNColumns = InteractionColumns

if __name__ == '__main__':
    print("线性模块：", len(LinerColumns))
    print(LinerColumns)

    print("DNN和交互模块:", len(InteractionColumns))
    print(InteractionColumns)

    # feature columns 处理函数校验

