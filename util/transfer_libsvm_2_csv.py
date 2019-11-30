#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: transfer_libsvm_2_csv.py
@time: 2019/11/28
"""
from util.load_dict import load_dict
from config.file_path_config import *
from config.featue_config import *
from codecs import open
import time


def clean_feature_map(feature_map_file=feature_map_file):
    feature_map_dict = load_dict(feature_map_file, key_first=True)
    feature_set = set()
    for key, value in feature_map_dict.items():
        has_merge_feature = False
        for f in merge_feature_ls:
            if f in key:
                feature_set.add(f)
                has_merge_feature = True
                break
        if has_merge_feature:
            continue
        if key.split("_")[-1] in feature_map_ignore_ls:
            key_base = key
        else:
            key_base = "_".join(key.split("_")[:-1])
        if not key_base:
            continue
        feature_set.add(key_base)

    for f in sorted(feature_set):
        print("\"{}\",".format(f))

    return feature_set


def transfer_libsvm_2_csv_file(raw_feature_file=test_raw_data_file, csv_file=test_csv_file):
    time1 = time.time()
    feature_base_ls = xgb_extract_feature
    with open(raw_feature_file, 'r', 'utf-8') as fin, open(csv_file, 'w', 'utf-8') as fout:
        for line in fin:
            label = line[0].strip()
            k_v_pair_ls = line[1:].strip().split(" ")
            k_v_pair_dict = {}
            for pair in k_v_pair_ls:
                # 是否有要合并的特征
                has_merge_feature = False
                for f in merge_feature_ls:
                    if f in pair:
                        v = pair.split(":")[0].replace(f + "_", "")
                        k_v_pair_dict[f] = v
                        has_merge_feature = True
                        break
                if has_merge_feature:
                    continue

                arr = pair.split("_")
                if len(arr) == 1:
                    continue
                k = "_".join(arr[:-1])
                v = arr[-1]
                if ":" not in v:
                    continue
                a1, a2 = v.split(":")
                if a1 in feature_map_ignore_ls:
                    k = "{}_{}".format(k, a1)
                    v = a2
                else:
                    v_1, v_2 = v.split(":")

                    if v_2 == "1":
                        v = v_1
                    else:
                        v = v_2
                if k not in k_v_pair_dict:
                    k_v_pair_dict[k] = v
                else:
                    # print(k,k_v_pair_dict[k],v)
                    k_v_pair_dict[k] += " %s" % v

            feature_value_ls = [k_v_pair_dict.get(x, "") for x in feature_base_ls]
            csv_str = ",".join([label] + feature_value_ls)
            fout.write(csv_str + "\n")
    print("timeout:{} min,  文件处理完成:{}".format((time.time() - time1) / 60, raw_feature_file))


def test_sample_csv_file(csv_file=test_csv_file):
    import pandas as pd
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)

    df = pd.read_csv(csv_file)
    df.columns = ["label"] + xgb_extract_feature

    print(df.describe())

    print(df.isna().sum(axis=0))

    for f in merge_feature_ls:
        print(f, "\n", df[f].value_counts(dropna=False))


def batch_transfer_libsvm_2_csv_file(raw_feature_file=test_raw_data_file, csv_file=test_csv_file, n_threads=20):
    from util.batch_process_file import batch_process_file
    batch_process_file(transfer_libsvm_2_csv_file, raw_feature_file, csv_file, n_threads=n_threads)


if __name__ == '__main__':
    # clean_feature_map()
    # transfer_libsvm_2_csv_file()
    # test_sample_csv_file()
    import sys

    n_threads = int(sys.argv[1])
    # batch_transfer_libsvm_2_csv_file(raw_feature_file=train_raw_data_file, csv_file=train_csv_file, n_threads=n_threads)
    # batch_transfer_libsvm_2_csv_file(raw_feature_file=test_raw_data_file, csv_file=test_csv_file, n_threads=n_threads)

    transfer_libsvm_2_csv_file(raw_feature_file=train_raw_data_file, csv_file=train_csv_file)
    # transfer_libsvm_2_csv_file(raw_feature_file=test_raw_data_file, csv_file=test_csv_file)