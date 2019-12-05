#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: sample_train_data_2_check_eda.py
@time: 2019/12/02
"""
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import codecs
from config.file_path_config import *
from config.deep_feature_config import *

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)
sns.set(style="whitegrid")

cate_features = CATEGORY_FEATURES
cont_features = CONTINUOUS_FEATURES
csv_features = FEATURE_NAMES

csv_file = test_csv_file


def load_data_2_df(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = csv_features

    feature_ls = list([(x, 'category') for x in cate_features] + [(x, float) for x in cont_features])
    dtypes = dict(feature_ls)
    df = df.astype(dtypes)
    return df


def eda_for_cate_features(df):
    n = len(df)
    for feature in cate_features:
        feature_value = df[feature]
        print("\n[%s]" % feature)
        print("缺失率：%.0f%%" % (feature_value.isna().sum() / n * 100))
        value_counts = feature_value.value_counts()
        print("value_counts:{}".format(len(value_counts)))
        print(value_counts[:50])
        if len(value_counts) < 20:
            sns.countplot(x=feature_value)
            plt.show()
        else:
            sns.violinplot(x=value_counts.values)
            plt.show()


def eda_for_cont_features(df):
    pass


def main():
    df = load_data_2_df(csv_file=csv_file)
    eda_for_cate_features(df)


def sample_data(total_data_path, sample_data_path, sample_nums):
    total_line_num = int(os.popen("wc -l %s" % total_data_path).readlines()[0].strip().split(" ")[0])
    assert sample_nums < total_line_num, "sample_nums<total_line_num!"
    sample_threshold = (sample_nums + 10) / total_line_num

    cnt = 0
    with codecs.open(total_data_path, 'r', 'utf-8') as fin, \
            codecs.open(sample_data_path, 'w', 'utf-8') as fout:
        for i, line in enumerate(fin):
            if i % 1000 == 0:
                print(i)
            if np.random.random() < sample_threshold:
                fout.write(line)
                cnt += 1
                if cnt >= sample_nums:
                    break
    sample_line_num = int(os.popen("wc -l %s" % sample_data_path).readlines()[0].strip().split(" ")[0])
    print("data sample completed! sample line num:{}. {}".format(sample_line_num, sample_data_path))


if __name__ == '__main__':
    import sys
    sample_num = 1000

    try:
        sample_num = int(sys.argv[1])
        print(sample_num)
    except Exception as e:
        print(e)
    print("final sample_num:", sample_num)
    sample_data(train_raw_data_file, train_raw_data_file + "_sample_%s" % sample_num, sample_num)
