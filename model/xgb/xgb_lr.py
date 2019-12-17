#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: test_xgb_lr.py
@time: 2019/12/16
"""
import sys
import os
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc


def test_lr(model_file, train_libsvm_file, test_libsvm_file):
    model = xgb.Booster(model_file=model_file)
    dtrain = xgb.DMatrix(train_libsvm_file)
    y_train_true = dtrain.get_label()
    x_train_lr = model.predict(dtrain, pred_leaf=True)
    df_train_lr = pd.DataFrame(x_train_lr).astype(str)

    lm = LogisticRegression(penalty=penalty, C=C)  # logestic model construction
    lm.fit(df_train_lr, y_train_true)  # fitting the data

    dtest = xgb.DMatrix(test_libsvm_file)
    y_test_true = dtest.get_label()
    model = xgb.Booster(model_file=model_file)
    x_test_lr = model.predict(dtest, pred_leaf=True)
    df_test_lr = pd.DataFrame(x_test_lr).astype(str)
    y_pred_test = lm.predict_proba(df_test_lr)
    y_pred_test = y_pred_test[:, 1]

    fpr, tpr, _ = roc_curve(y_test_true, y_pred_test, pos_label=1)
    auc_score = auc(fpr, tpr)

    exp_result = "auc:%.4f" % auc_score
    print(exp_result)



def main():
    model_file = os.path.join(data_path, "model_file")
    train_libsvm_file = os.path.join(data_path, "train_libsvm_feature")
    test_libsvm_file = os.path.join(data_path, "test_libsvm_feature")
    test_lr(model_file, train_libsvm_file, test_libsvm_file)


if __name__ == '__main__':
    data_path = sys.argv[1]
    penalty = sys.argv[2]
    C = float(sys.argv[3])
    main()