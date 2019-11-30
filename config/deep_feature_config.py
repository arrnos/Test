# -*- coding: utf-8 -*-

"""
@author: zhangmeng
@file: deep_feature_config.py
@time: 2019/12/10 17:17
"""

import tensorflow as tf

from config.file_path_config import *

FEATURE_INFOS = [
    ["account_age", int, 0, 1],
    ["alignment_day", str, None, 1],
    ["alignment_day_of_week", str, None, 1],
    ["alignment_hour", str, None, 1],
    ["alignment_month", str, None, 1],
    ["alignment_year", str, None, 1],
    ["all_call_num", int, 0, 1],
    ["avgerage_waiting_time_of_dialogue", float, 0, 1],
    ["consult_type", str, None, 1],
    ["contain_education_or_promotion_key_word_in_dialogue", str, None, 1],
    ["contain_price_key_word_in_dialogue", str, None, 1],
    ["create_user", str, None, 1],
    ["delta_days_from_entryDate", int, 0, 1],
    ["delta_time_of_create_time_and_operator_time", float, 0, 1],
    ["dialogue_start_at", str, None, 1],
    ["duration_of_dialogue", float, 0, 1],
    ["educationalBackground", str, None, 1],
    ["first_proj_id", str, None, 1],
    ["gender", str, None, 1],
    ["graduateSchool", str, None, 1],
    ["major", str, None, 1],
    ["marriageState", str, None, 1],
    ["message_opp_distribution_num", int, 0, 1],
    ["message_opp_distribution_saturation", float, 0, 1],
    ["message_opp_limit_num", int, 0, 1],
    ["nationalEdu", str, None, 1],
    ["number_of_dialogue", int, 0, 1],
    ["number_of_student_dialogue", int, 0, 1],
    ["online_opp_distribution_num", int, 0, 1],
    ["online_opp_distribution_saturation", float, 0, 1],
    ["online_opp_limit_num", int, 0, 1],
    ["opp_completed_rate", float, 0, 1],
    ["opp_following_num", int, 0, 1],
    ["opp_today_following_num", int, 0, 1],
    ["oppor_source", str, None, 1],
    ["pastNday_advertiser_applied_ratio", float, 0, 1],
    ["pastNday_advertiser_distribution_num", int, 0, 1],
    ["pastNday_advertiser_order_num", int, 0, 1],
    ["pastNday_applied_ratio_on_siteId", float, 0, 1],
    ["pastNday_distribution_num_on_siteId", int, 0, 1],
    ["pastNday_order_num_on_siteId", int, 0, 1],
    ["past_n_days_acc_alignment_num", int, 0, 1],
    ["past_n_days_acc_order_amount", int, 0, 1],
    ["past_n_days_acc_order_avg_amount", int, 0, 1],
    ["past_n_days_acc_order_num", int, 0, 1],
    ["past_n_days_acc_order_rate", float, 0, 1],
    ["posLevel", str, None, 1],
    ["promotion_type", str, None, 1],
    ["quantum_id", str, None, 1],
    ["race", str, None, 1],
    ["recycle_opp_distribution_num", int, 0, 1],
    ["recycle_opp_distribution_saturation", float, 0, 1],
    ["recycle_opp_limit_num", int, 0, 1],
    ["residenceType", str, None, 1],
    ["site_source", str, None, 1],
    ["student_cellphone_midnum", str, None, 1],
    ["student_dialogue_fenci", str, None, 0],
    ["timedelta_from", str, None, 1],
    ["today_unfollowed_num", int, 0, 1],
    ["twice_consult", str, None, 1],
    ["valid_call_num", int, 0, 1],
]


def parse_feature_config():
    feature_name_list = []
    feature_dtype_list = []
    feature_default_list = []
    feature_keras_input_dict = {}
    for arr in FEATURE_INFOS:
        f_name, dtype, default, use = arr
        if use != 1:
            continue
        feature_name_list.append(f_name)
        feature_dtype_list.append(dtype)
        if dtype == str:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.string)
            feature_default_list.append(str(default))
        elif dtype == int:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.int64)
            feature_default_list.append(int(default))
        elif dtype == float:
            feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=tf.float32)
            feature_default_list.append(float(default))
    return feature_name_list, feature_dtype_list, feature_default_list, feature_keras_input_dict


FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_KERAS_INPUT_DICT = parse_feature_config()

if __name__ == '__main__':
    for i in [FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_KERAS_INPUT_DICT]:
        print(i)
