#!usrbinenv python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: file_path_config.py
@time: 20191128
"""
import os
import platform

plat = platform.system().lower()
assert plat in ("linux", "windows", "darwin"), "platform is not linux or windows or mac!"

PROJECT_NAME = "Test"

if plat == "linux":
    user = "zhangmeng"
    data_path = os.path.join("/home/%s/" % user, "project_data", PROJECT_NAME)
    project_path = os.path.join("/home/%s/" % user, PROJECT_NAME)
elif plat == "windows":
    data_path = os.path.join("E:\project_data", PROJECT_NAME)
    project_path = os.path.dirname(os.getcwd())
else:
    user = "arrnos"
    data_path = os.path.join("/Users/", user, "project_data", PROJECT_NAME)
    project_path = os.path.dirname(os.getcwd())

assert os.path.isdir(data_path), "%s不存在！" % data_path

middle_path = os.path.join(data_path, "middle_file")
xgb_path = os.path.join(data_path, "xgb")
feature_map_file = os.path.join(xgb_path, "train_feature_map")
feature_info_file = os.path.join(project_path, "config", "feature_info.csv")

if plat == "linux":
    train_libsvm_file = os.path.join(data_path, "train_libsvm_feature")
    test_libsvm_file = os.path.join(data_path, "test_libsvm_feature")
    train_raw_data_file = os.path.join(data_path, "merged_raw_train_feature")
    test_raw_data_file = os.path.join(data_path, "merged_raw_test_feature")
    train_csv_file = os.path.join(data_path, "train_csv_feature")
    test_csv_file = os.path.join(data_path, "test_csv_feature")
else:
    train_libsvm_file = os.path.join(data_path, "train_libsvm_feature_sample")
    test_libsvm_file = os.path.join(data_path, "test_libsvm_feature_sample")
    train_raw_data_file = os.path.join(data_path, "merged_raw_train_feature_sample")
    test_raw_data_file = os.path.join(data_path, "merged_raw_test_feature_sample")
    train_csv_file = os.path.join(data_path, "train_csv_feature_sample")
    test_csv_file = os.path.join(data_path, "test_csv_feature_sample")

if __name__ == '__main__':
    print(os.path.exists(data_path))
    print(test_raw_data_file)
