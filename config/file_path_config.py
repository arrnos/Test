#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: file_path_config.py
@time: 2019/11/28
"""
import os

data_path = os.path.abspath("data/")
middle_path = data_path+"/middle_file/"
xgb_path = data_path + "/xgb/"

train_libsvm_file = data_path + "/train_libsvm_feature"
test_libsvm_file = data_path + "/test_libsvm_feature"
train_raw_data_file = data_path + "/merged_raw_train_feature"
test_raw_data_file = data_path + "/merged_raw_test_feature"

if __name__ == '__main__':
    print(os.path.exists(data_path))
    print(test_raw_data_file)