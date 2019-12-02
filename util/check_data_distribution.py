#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: check_data_distribution.py
@time: 2019/12/02
"""
from config.deep_feature_config import *
from config.xgb_feature_config import *
import pandas as pd
import seaborn

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


def show_data_info(csv_file=test_csv_file):
    df = pd.read_csv(csv_file)
    df.columns = ["label"] + xgb_extract_feature

    print(df.describe())

    print(df.isna().sum(axis=0))

    for f in merge_feature_ls:
        print(f, "\n", df[f].value_counts(dropna=False))

if __name__ == '__main__':
    show_data_info()
    print(train_csv_file)
    print(xgb_extract_feature)