#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: load_dict.py
@time: 2019/11/28
"""
import codecs


def load_dict(file_name, splitor="\t",key_first=True):
    dict_return = {}
    with codecs.open(file_name, 'r', "utf-8") as fin:
        for line in fin:
            arr = line.split(splitor)
            if len(arr) != 2:
                continue
            if key_first:
                dict_return[arr[0].strip()] = arr[1].strip()
            else:
                dict_return[arr[1].strip()] = arr[0].strip()

    return dict_return

