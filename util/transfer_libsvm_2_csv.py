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


def clean_feature_map(feature_map_file=feature_map_file):
    feature_map_dict = load_dict(feature_map_file, key_first=True)
    feature_set = set()
    for key, value in feature_map_dict.items():
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


def transfer_libsvm_2_csv_file(raw_feature_file=test_raw_data_file, csv_file=test_csv_file,
                               feature_base_ls=xgb_libsvm_feature):
    with open(raw_feature_file, 'r', 'utf-8') as fin, open(csv_file, 'w', 'utf-8') as fout:
        for line in fin:
            label = line[0].strip()
            k_v_pair_ls = line[1:].strip().split(" ")
            k_v_pair_dict = {}
            for pair in k_v_pair_ls:
                arr = pair.split("_")
                if len(arr) ==1:
                    continue
                k = "_".join(arr[:-1])
                v = arr[-1]
                a1,a2 = v.split(":")
                # consult不再？？
                if a1 in feature_map_ignore_ls:
                    k= "{}_{}".format(k,a1)
                    v= a2

                if ":" in v:
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


if __name__ == '__main__':
    transfer_libsvm_2_csv_file()
    # clean_feature_map()