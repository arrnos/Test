#!usrbinenv python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: file_path_config.py
@time: 20191128
"""
import os

data_path = os.path.join(os.path.dirname(os.getcwd()),"data")
middle_path = os.path.join(data_path, "middle_file")
xgb_path = os.path.join(data_path, "xgb")

# train_libsvm_file = os.path.join(data_path , "train_libsvm_feature")
# test_libsvm_file = os.path.join(data_path , "test_libsvm_feature")
# train_raw_data_file = os.path.join(data_path , "merged_raw_train_feature")
# test_raw_data_file = os.path.join(data_path , "merged_raw_test_feature")
# train_csv_file = os.path.join(data_path, "train_csv_feature")
# test_csv_file = os.path.join(data_path, "test_csv_feature")

train_libsvm_file = os.path.join(data_path, "train_libsvm_feature_sample")
test_libsvm_file = os.path.join(data_path, "test_libsvm_feature_sample")
train_raw_data_file = os.path.join(data_path, "merged_raw_train_feature_sample")
test_raw_data_file = os.path.join(data_path, "merged_raw_test_feature_sample")
train_csv_file = os.path.join(data_path, "train_csv_feature_sample")
test_csv_file = os.path.join(data_path, "test_csv_feature_sample")

feature_map_file = os.path.join(xgb_path, "train_feature_map")


if __name__ == '__main__':
    print(os.path.exists(data_path))
    print(test_raw_data_file)
