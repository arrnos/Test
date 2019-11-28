#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: model_config.py
@time: 2019/11/28
"""
from config.file_path_config import *

xgb_config = {
    'num_round': 100,
    "param":
        {
            'nthread': 6,
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.3,
            'silent': 1,
            'objective': 'binary:logistic',
            'tree_method': 'exact'
        },

    "train_libsvm_file": train_libsvm_file,
    "test_libsvm_file": test_libsvm_file,
    "model_file": xgb_path + "model_file",
    "dump_file": xgb_path + "dump_file",
    "dump_nice_file": xgb_path + "dump_nice_file",
    "feature_map_file": xgb_path + "train_feature_map",
    "feature_importance_file": xgb_path + "feature_importance_file",
    "exp_result_file": xgb_path + "exp_result_file"
}
