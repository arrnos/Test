# -*- coding: utf-8 -*-

"""
@author: zhangmeng
@file: deep_feature_config.py
@time: 2019/12/10 17:17
"""

import tensorflow as tf

from config.file_path_config import *

FEATURE_INFOS = [
    ["label", tf.int64, 0, 1],

    ["account_age", tf.int64, 0, 1],
    ["alignment_day", tf.string, "", 1],
    ["alignment_day_of_week", tf.string, "", 1],
    ["alignment_hour", tf.string, "", 1],
    ["alignment_month", tf.string, "", 1],
    ["alignment_year", tf.string, "", 1],
    ["all_call_num", tf.float32, 0.0, 1],
    ["avgerage_waiting_time_of_dialogue", tf.float32, 0.0, 1],
    ["consult_type", tf.string, "", 1],
    ["contain_education_or_promotion_key_word_in_dialogue", tf.string, "", 1],
    ["contain_price_key_word_in_dialogue", tf.string, "", 1],
    ["create_user", tf.string, "", 1],
    ["delta_days_from_entryDate", tf.int64, 0, 1],
    ["delta_time_of_create_time_and_operator_time", tf.float32, 0.0, 1],
    ["dialogue_start_at", tf.string, "", 1],
    ["duration_of_dialogue", tf.float32, 0.0, 1],
    ["educationalBackground", tf.string, "", 1],
    ["first_proj_id", tf.string, "", 1],
    ["gender", tf.string, "", 1],
    ["graduateSchool", tf.string, "", 1],
    ["major", tf.string, "", 1],
    ["marriageState", tf.string, "", 1],
    ["message_opp_distribution_num", tf.int64, 0, 1],
    ["message_opp_distribution_saturation", tf.float32, 0.0, 1],
    ["message_opp_limit_num", tf.int64, 0, 1],
    ["nationalEdu", tf.string, "", 1],
    ["number_of_dialogue", tf.int64, 0, 1],
    ["number_of_student_dialogue", tf.int64, 0, 1],
    ["online_opp_distribution_num", tf.int64, 0, 1],
    ["online_opp_distribution_saturation", tf.float32, 0.0, 1],
    ["online_opp_limit_num", tf.int64, 0, 1],
    ["opp_completed_rate", tf.float32, 0.0, 1],
    ["opp_following_num", tf.int64, 0, 1],
    ["opp_today_following_num", tf.int64, 0, 1],
    ["oppor_source", tf.string, "", 1],
    ["pastNday_advertiser_applied_ratio", tf.float32, 0.0, 1],
    ["pastNday_advertiser_distribution_num", tf.int64, 0, 1],
    ["pastNday_advertiser_order_num", tf.int64, 0, 1],
    ["pastNday_applied_ratio_on_siteId", tf.float32, 0.0, 1],
    ["pastNday_distribution_num_on_siteId", tf.int64, 0, 1],
    ["pastNday_order_num_on_siteId", tf.int64, 0, 1],
    ["past_n_days_acc_alignment_num", tf.int64, 0, 1],
    ["past_n_days_acc_order_amount", tf.float32, 0.0, 1],
    ["past_n_days_acc_order_avg_amount", tf.float32, 0.0, 1],
    ["past_n_days_acc_order_num", tf.int64, 0, 1],
    ["past_n_days_acc_order_rate", tf.float32, 0.0, 1],
    ["posLevel", tf.string, "", 1],
    ["promotion_type", tf.string, "", 1],
    ["quantum_id", tf.string, "", 1],
    ["race", tf.string, "", 1],
    ["recycle_opp_distribution_num", tf.int64, 0, 1],
    ["recycle_opp_distribution_saturation", tf.float32, 0.0, 1],
    ["recycle_opp_limit_num", tf.int64, 0, 1],
    ["residenceType", tf.string, "", 1],
    ["site_source", tf.string, "", 1],
    ["student_cellphone_midnum", tf.string, "", 1],
    ["student_dialogue_fenci", tf.string, "", 0],
    ["timedelta_from", tf.float32, 0.0, 1],
    ["today_unfollowed_num", tf.int64, 0, 1],
    ["twice_consult", tf.string, "", 1],
    ["valid_call_num", tf.int64, 0, 1],
]


def parse_feature_config():
    feature_name_list = []
    feature_dtype_list = []
    feature_default_list = []
    feature_use_list = []
    feature_name_use_list = []
    category_feature_list = []
    continuous_feature_list = []

    feature_keras_input_dict = {}
    for i, arr in enumerate(FEATURE_INFOS):
        f_name, dtype, default, use = arr
        feature_name_list.append(f_name)
        feature_dtype_list.append(dtype)

        assert dtype in [tf.string, tf.int64, tf.float32]

        if use == 1:
            feature_use_list.append(i)
            feature_name_use_list.append(f_name)
            feature_default_list.append(default)

        if use != 1 or f_name == "label":
            continue

        feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=dtype)

        if dtype == tf.string:
            category_feature_list.append(f_name)
        else:
            continuous_feature_list.append(f_name)

    return feature_name_list, feature_dtype_list, feature_default_list, feature_use_list, feature_name_use_list, category_feature_list, continuous_feature_list, feature_keras_input_dict


FEATURE_NAMES, FEATURE_DTYPES, \
FEATURE_DEFAULTS, FEATURE_USE_LIST, FEATURE_NAME_USE_LIST, \
CATEGORY_FEATURES, CONTINUOUS_FEATURES, \
FEATURE_KERAS_INPUT_DICT = parse_feature_config()

# 参数初步校验
assert len(FEATURE_NAMES) == len(FEATURE_DTYPES)
assert len(FEATURE_USE_LIST) == len(FEATURE_DEFAULTS) == len(FEATURE_NAME_USE_LIST)
assert len(CATEGORY_FEATURES) + len(CONTINUOUS_FEATURES) == len(FEATURE_KERAS_INPUT_DICT)
assert len(FEATURE_NAME_USE_LIST) == len(FEATURE_KERAS_INPUT_DICT) + 1


# 从csv文件读取dataset

def _parse_line(line):
    fields = tf.io.decode_csv(line, FEATURE_DEFAULTS, select_cols=FEATURE_USE_LIST)
    features = dict(zip(FEATURE_NAME_USE_LIST, fields))
    label = features.pop("label")
    return features, label


def read_csv_2_dataset(csv_file, shuffle_size=10000, batch_size=256):
    data = tf.data.TextLineDataset(csv_file)
    data = data.shuffle(shuffle_size).map(_parse_line).batch(batch_size)
    return data


if __name__ == '__main__':
    # for i in [FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_KERAS_INPUT_DICT]:
    #     print(i)

    print(CATEGORY_FEATURES)
    print(CONTINUOUS_FEATURES)
    print(FEATURE_NAMES)

    # dataset = read_csv_2_dataset(test_csv_file, shuffle_size=10000, batch_size=256)
    # cnt = 0
    # for d in dataset:
    #     cnt += 1
    #     print(d[1])
    #     for k, v in d[0].items():
    #         print(k)
    #         print(v)
    #     if cnt > 1:
    #         break
