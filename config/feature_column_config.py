#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: feature_column_config.py
@time: 2019/11/11
"""

from itertools import chain

from tensorflow import feature_column as fc
from tensorflow.python.keras.initializers import RandomNormal
from util.cate_values_util import read_feature_values_from_file
from util.cate_values_util import read_feature_values_from_file_limit

from util.dense_values_util import *

SPARSE_EMBEDDING_SIZE = 8
BUCKTE_EMBEDDING_SIZE = 8

# embedding是否需要加入正则
EMBED_L2_REG = 1
embed_init_std = 0.001
embed_l2_reg = 0.0005

# 离散值values是否需要进行数量限制
SPARSE_VALUES_LIMIT = 1
read_feature_values = read_feature_values_from_file_limit if SPARSE_VALUES_LIMIT else read_feature_values_from_file

# 连续变量的归一化后的分桶边界
dense_bound_list = [x / 10 for x in range(1, 10)]

# 连续变量的归一化方式
MIN_MAX_DICT = load_min_max_value_dict_from_file(min_max_value_path)
LOG_MIN_MAX_DICT = prepare_log_min_max_dict(MIN_MAX_DICT)


# def min_max_norm_func(f):
#     min_value, max_value = MIN_MAX_DICT[f]
#     func = lambda x: 1.0 * (tf.cast(x, float) - min_value) / (max_value - min_value)
#     return func

def min_max_norm_func(f):
    min_value, max_value = MIN_MAX_DICT[f]

    def func(x):
        x_float = tf.cast(x, tf.float32)
        result = tf.where(x_float >= max_value, 1.0,
                          1.0 * (x_float - min_value) / (max_value - min_value))
        return result

    return func


def log_min_max_func(f):
    min_value, max_value = LOG_MIN_MAX_DICT[f]

    def func(x):
        x_float = tf.math.log1p(tf.cast(x, tf.float32))
        result = tf.where(x_float >= max_value, 1.0,
                          1.0 * (x_float - min_value) / (max_value - min_value))
        return result

    return func


dense_process_dict = dict(
    [(f, log_min_max_func(f)) for f in LOG_MIN_MAX_METHOD_LIST] +
    [(f, min_max_norm_func(f)) for f in MIN_MAX_METHOD_LIST])

# =============================连续特征=======================================

# dense feature
dense_features = {f_name: fc.numeric_column(f_name, normalizer_fn=dense_process_dict[f_name]) for f_name in
                  CONTINUOUS_FEATURE_USE_LIST}
# dense bucket
dense_features_bucket = {f_name: fc.bucketized_column(column, dense_bound_list) for f_name, column in
                         dense_features.items()}
# dense embedding
if not EMBED_L2_REG:
    dense_features_emb = {f_name: fc.embedding_column(column, BUCKTE_EMBEDDING_SIZE) for f_name, column in
                          dense_features_bucket.items()}
else:
    dense_features_emb = {f_name: fc.embedding_column(column,
                                                      BUCKTE_EMBEDDING_SIZE,
                                                      initializer=RandomNormal(mean=0.0, stddev=embed_init_std,
                                                                               seed=1024),
                                                      max_norm=embed_l2_reg
                                                      ) for f_name, column in dense_features_bucket.items()}

# ===========================离散特征===============================

# sparse feature
sparse_features = \
    {f_name: fc.categorical_column_with_vocabulary_list(f_name, read_feature_values(f_name), num_oov_buckets=10) for
     f_name in CATEGORY_FEATURE_USE_LIST}

# sparse feature one_hot
sparse_features_one_hot = \
    {f_name: fc.indicator_column(column) for f_name, column in sparse_features.items()}

# sparse feature embedding
if not EMBED_L2_REG:
    sparse_features_emb = \
        {f_name: fc.embedding_column(column, SPARSE_EMBEDDING_SIZE) for f_name, column in sparse_features.items()}
else:
    sparse_features_emb = \
        {f_name: fc.embedding_column(column,
                                     SPARSE_EMBEDDING_SIZE,
                                     initializer=RandomNormal(mean=0.0, stddev=embed_init_std, seed=1024),
                                     max_norm=embed_l2_reg)
         for f_name, column in sparse_features.items()}

# ========================= LR feature column =============================

LinerColumns = [
    sparse_features_one_hot.values(),
    dense_features.values()
]

LinerColumns = list(chain(*LinerColumns))

# ========================= FM feature column =============================
InteractionColumns = [
    sparse_features_emb.values(),
    dense_features_emb.values()
]

InteractionColumns = list(chain(*InteractionColumns))

# ========================= DNN feature column =============================

DNNColumns = InteractionColumns


# ==========================   Test module =================================
def get_test_tensor():
    example = {"label": [1], "account_age": [27, 88], "alignment_day": ["22"], "alignment_day_of_week": ["2"],
               "alignment_hour": [12],
               "alignment_month": ["9"], "alignment_year": ["2019", "2018"], "all_call_num": [22],
               "avgerage_waiting_time_of_dialogue": [12], "opp_today_following_num": [5],
               "consult_type": ["ALL"], "contain_education_or_promotion_key_word_in_dialogue": [""],
               "contain_price_key_word_in_dialogue": ["1"], "create_user": ["system"],
               "delta_days_from_entryDate": [120, 10000],
               "delta_time_of_create_time_and_operator_time": [100], "dialogue_start_at": ["noon"],
               "duration_of_dialogue": [13], "educationalBackground": ["1"], "first_proj_id": ["53"],
               "gender": ["M"], "graduateSchool": ["武汉职业技术学院"], "major": ["市场营销"],
               "marriageState": ["1"], "message_opp_distribution_num": [10],
               "message_opp_distribution_saturation": [0.1],
               "message_opp_limit_num": [20], "nationalEdu": ["1"], "number_of_dialogue": [7],
               "number_of_student_dialogue": [3], "online_opp_distribution_num": [18],
               "online_opp_distribution_saturation": [0.3],
               "online_opp_limit_num": [60], "opp_completed_rate": [0.2], "opp_following_num": [7],
               "oppor_source": ["ONLINE_CS_GREATBEAR"], "pastNday_advertiser_applied_ratio": [0.1],
               "pastNday_advertiser_distribution_num": [5000], "pastNday_advertiser_order_num": [321],
               "pastNday_applied_ratio_on_siteId": [0.05], "pastNday_distribution_num_on_siteId": [100],
               "pastNday_order_num_on_siteId": [123], "past_n_days_acc_alignment_num": [123],
               "past_n_days_acc_order_amount": [100000], "past_n_days_acc_order_avg_amount": [10000],
               "past_n_days_acc_order_num": [5], "past_n_days_acc_order_rate": [0.04],
               "posLevel": ["1"], "promotion_type": ["SEM"], "quantum_id": ["2030006"],
               "race": ["汉"], "recycle_opp_distribution_num": [10], "recycle_opp_distribution_saturation": [0.05],
               "recycle_opp_limit_num": [12], "residenceType": ["1"], "site_source": ["微信朋友圈移动平台"],
               "student_cellphone_midnum": [""],  # "student_dialogue_fenci":["好"],
               "timedelta_from": [2], "today_unfollowed_num": [12], "twice_consult": [""], "valid_call_num": [12]}

    example_tensor = {x: tf.constant(y) for x, y in example.items()}
    return example_tensor


def valid_columns():
    example_tensor = get_test_tensor()

    test_dense_feature_columns = [
        (f_name, (dense_features[f_name], dense_features_bucket[f_name], dense_features_emb[f_name]))
        for f_name in test_dense_features]
    for f_name, (sparse, one_hot, emb) in test_dense_feature_columns:
        print(f_name, example_tensor[f_name].numpy())
        print(MIN_MAX_DICT[f_name])
        print(LOG_MIN_MAX_DICT[f_name])

        print(tf.keras.layers.DenseFeatures([sparse])(example_tensor))
        print(tf.keras.layers.DenseFeatures([one_hot])(example_tensor))
        print(tf.keras.layers.DenseFeatures([emb])(example_tensor))
        print()

    test_sparse_feature_columns = [
        (f_name, (sparse_features[f_name], sparse_features_one_hot[f_name], sparse_features_emb[f_name]))
        for f_name in test_sparse_features]
    for f_name, (sparse, one_hot, emb) in test_sparse_feature_columns:
        print(f_name, example_tensor[f_name].numpy())

        print(tf.keras.layers.DenseFeatures([one_hot])(example_tensor))
        print(tf.keras.layers.DenseFeatures([emb])(example_tensor))
        print()


if __name__ == '__main__':
    print("线性模块：", len(LinerColumns))
    print(LinerColumns)

    print("DNN和交互模块:", len(InteractionColumns))
    print(InteractionColumns)
    print()

    test_dense_features = ["delta_days_from_entryDate", "account_age", "timedelta_from"]
    test_sparse_features = ["posLevel", "alignment_day", "alignment_year"]
    valid_columns()
