#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: train.py
@time: 2019/11/28
"""
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from config.model_config import xgb_config as config
from util.load_dict import *
import re
import json
import datetime


def train(train_libsvm_file, dump_file, model_file):
    dtrain = xgb.DMatrix(train_libsvm_file)
    watch_ls = [(dtrain, 'eval_train')]
    bst = xgb.train(config["param"], dtrain, config["num_round"], watch_ls)
    bst.dump_model(dump_file)
    bst.save_model(model_file)


def test(test_libsvm_file, model_file, exp_result_file):
    model = xgb.Booster(model_file=model_file)
    dtest = xgb.DMatrix(test_libsvm_file)
    y_true = dtest.get_label()
    y_pred = model.predict(dtest)

    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auc1 = auc(fpr, tpr)
    auc2 = roc_auc_score(y_true, y_pred)
    auc3 = roc_auc_score(y_true, y_pred, average=None)
    prc = precision_recall_curve(y_true, y_pred, pos_label=1)

    exp_result = "auc1:%.4f\tauc2:%.4f\tauc3:%.4f\tprc:%.4f" % (auc1, auc2, auc3, prc)
    print(exp_result)

    with codecs.open(exp_result_file, 'w', 'utf-8') as fout:
        fout.write("[exp_result]\n" + exp_result + "\n\n")
        fout.write(json.dumps(config, indent=2))


def print_nice_model(model_dump_file, feature_map_file, model_dump_nice_file):
    dict_feature_index = load_dict(feature_map_file)
    pattern = re.compile("f[0-9]+")
    with codecs.open(model_dump_file, 'r', 'utf-8') as fin, codecs.open(model_dump_nice_file, 'w', 'utf-8') as fout:
        for line in fin:
            matcher = pattern.search(line)
            if matcher is None:
                fout.write(line)
                continue
            feature_index = "%s" % (int(matcher.group()[1:]))
            name = dict_feature_index[feature_index]
            line = line.replace(matcher.group(), name)
            fout.write(line)


def get_feature_importance(model_file, feature_map_file, out_put_file, importance_metric="weight"):
    bst = xgb.Booster(model_file=model_file)
    feature_importance = bst.get_score(importance_type=importance_metric)
    importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    dict_featureName = load_dict(feature_map_file, "\t", key_first=False)

    with codecs.open(out_put_file, 'w', 'utf-8') as fout:
        fout.write("特征排名\t特征名称\t特征重要度\n")
        for index, (f_idx, score) in enumerate(importance):
            fout.write("\t".join([str(x) for x in [index, dict_featureName[str(f_idx)[1:]], score]]) + "\n")


def main():
    train_libsvm_file = config["train_libsvm_file"]
    dump_file = config["dump_file"]
    model_file = config["model_file"]
    test_libsvm_file = config["test_libsvm_file"]
    feature_map_file = config["feature_map_file"]
    dump_nice_file = config["dump_nice_file"]
    feature_importance_file = config["feature_importance_file"]
    exp_result_file = config["exp_result_file"] + "_%s" % datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")

    train(train_libsvm_file, dump_file, model_file)
    test(test_libsvm_file, model_file, exp_result_file)

    print_nice_model(dump_file, feature_map_file, dump_nice_file)
    get_feature_importance(model_file, feature_map_file, feature_importance_file, importance_metric="weight")
