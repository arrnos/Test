# -*- coding: utf-8 -*-

"""
@author: zhangmeng
@file: deep_feature_config.py
@time: 2019/12/10 17:17
"""

import tensorflow as tf

from config.file_path_config import *

FEATURE_INFOS = [
    ["label", int, 0, 1],

    ["account_age", int, 0, 1],
    ["alignment_day", str, "", 1],
    ["alignment_day_of_week", str, "", 1],
    ["alignment_hour", str, "", 1],
    ["alignment_month", str, "", 1],
    ["alignment_year", str, "", 1],
    ["all_call_num", float, 0.0, 1],
    ["avgerage_waiting_time_of_dialogue", float, 0.0, 1],
    ["consult_type", str, "", 1],
    ["contain_education_or_promotion_key_word_in_dialogue", str, "", 1],
    ["contain_price_key_word_in_dialogue", str, "", 1],
    ["create_user", str, "", 1],
    ["delta_days_from_entryDate", int, 0, 1],
    ["delta_time_of_create_time_and_operator_time", float, 0.0, 1],
    ["dialogue_start_at", str, "", 1],
    ["duration_of_dialogue", float, 0.0, 1],
    ["educationalBackground", str, "", 1],
    ["first_proj_id", str, "", 1],
    ["gender", str, "", 1],
    ["graduateSchool", str, "", 1],
    ["major", str, "", 1],
    ["marriageState", str, "", 1],
    ["message_opp_distribution_num", int, 0, 1],
    ["message_opp_distribution_saturation", float, 0.0, 1],
    ["message_opp_limit_num", int, 0, 1],
    ["nationalEdu", str, "", 1],
    ["number_of_dialogue", int, 0, 1],
    ["number_of_student_dialogue", int, 0, 1],
    ["online_opp_distribution_num", int, 0, 1],
    ["online_opp_distribution_saturation", float, 0.0, 1],
    ["online_opp_limit_num", int, 0, 1],
    ["opp_completed_rate", float, 0.0, 1],
    ["opp_following_num", int, 0, 1],
    ["opp_today_following_num", int, 0, 1],
    ["oppor_source", str, "", 1],
    ["pastNday_advertiser_applied_ratio", float, 0.0, 1],
    ["pastNday_advertiser_distribution_num", int, 0, 1],
    ["pastNday_advertiser_order_num", int, 0, 1],
    ["pastNday_applied_ratio_on_siteId", float, 0.0, 1],
    ["pastNday_distribution_num_on_siteId", int, 0, 1],
    ["pastNday_order_num_on_siteId", int, 0, 1],
    ["past_n_days_acc_alignment_num", int, 0, 1],
    ["past_n_days_acc_order_amount", float, 0.0, 1],
    ["past_n_days_acc_order_avg_amount", float, 0.0, 1],
    ["past_n_days_acc_order_num", int, 0, 1],
    ["past_n_days_acc_order_rate", float, 0.0, 1],
    ["posLevel", str, "", 1],
    ["promotion_type", str, "", 1],
    ["quantum_id", str, "", 1],
    ["race", str, "", 1],
    ["recycle_opp_distribution_num", int, 0, 1],
    ["recycle_opp_distribution_saturation", float, 0.0, 1],
    ["recycle_opp_limit_num", int, 0, 1],
    ["residenceType", str, "", 1],
    ["site_source", str, "", 1],
    ["student_cellphone_midnum", str, "", 1],
    ["student_dialogue_fenci", str, "", 0],
    ["timedelta_from", float, 0.0, 1],
    ["today_unfollowed_num", int, 0, 1],
    ["twice_consult", str, "", 1],
    ["valid_call_num", int, 0, 1],
]


def parse_feature_config():
    feature_name_list = []
    feature_dtype_list = []
    feature_default_list = []
    feature_use_list = []
    feature_name_use_list = []
    feature_keras_input_dict = {}
    for i, arr in enumerate(FEATURE_INFOS):
        f_name, dtype, default, use = arr
        feature_name_list.append(f_name)
        feature_dtype_list.append(dtype)
        if use == 1:
            feature_use_list.append(i)
            feature_name_use_list.append(f_name)
            feature_default_list.append(default)

        if use != 1 or f_name == "label":
            continue
        if dtype == str:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.string)
        elif dtype == int:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.int64)
        elif dtype == float:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.float32)
    return feature_name_list, feature_dtype_list, feature_default_list, feature_use_list, feature_name_use_list, feature_keras_input_dict


FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_USE_LIST, FEATURE_NAME_USE_LIST, FEATURE_KERAS_INPUT_DICT = parse_feature_config()


def _parse_line(line):
    fields = tf.io.decode_csv(line, FEATURE_DEFAULTS, select_cols=FEATURE_USE_LIST)
    features = dict(zip(FEATURE_NAME_USE_LIST, fields))
    label = features.pop("label")
    return features, label


def read_csv_2_dataset(csv_file, shuffle_size=10000,batch_size=256):
    data = tf.data.TextLineDataset(csv_file)
    data = data.shuffle(shuffle_size).map(_parse_line).batch(batch_size)
    return data


if __name__ == '__main__':
    # for i in [FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_KERAS_INPUT_DICT]:
    #     print(i)

    dataset = read_csv_2_dataset(test_csv_file, shuffle_size=10000,batch_size=256)
    cnt = 0
    for d in dataset:
        cnt+=1
        print(d[1])
        for k,v in d[0].items():
            print(k)
            print(v)
        if cnt>1:
            break
