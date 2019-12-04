#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: python3.7
@author: zhangmeng
@file: dump_category_values_2_file.py
@time: 2019/12/02
"""

# from config.deep_feature_config import *
from config.file_path_config import category_value_path
import os
from util.load_dict import load_dict
from codecs import open
from collections import defaultdict

CATEGORY_FEATURES = ['alignment_day', 'alignment_day_of_week', 'alignment_hour', 'alignment_month', 'alignment_year',
                     'consult_type', 'contain_education_or_promotion_key_word_in_dialogue',
                     'contain_price_key_word_in_dialogue', 'create_user', 'dialogue_start_at', 'educationalBackground',
                     'first_proj_id', 'gender', 'graduateSchool', 'major', 'marriageState', 'nationalEdu',
                     'oppor_source', 'posLevel', 'promotion_type', 'quantum_id', 'race', 'residenceType', 'site_source',
                     'student_cellphone_midnum', 'twice_consult']
FEATURE_NAMES = ['label', 'account_age', 'alignment_day', 'alignment_day_of_week', 'alignment_hour', 'alignment_month',
                 'alignment_year', 'all_call_num', 'avgerage_waiting_time_of_dialogue', 'consult_type',
                 'contain_education_or_promotion_key_word_in_dialogue', 'contain_price_key_word_in_dialogue',
                 'create_user', 'delta_days_from_entryDate', 'delta_time_of_create_time_and_operator_time',
                 'dialogue_start_at', 'duration_of_dialogue', 'educationalBackground', 'first_proj_id', 'gender',
                 'graduateSchool', 'major', 'marriageState', 'message_opp_ditf.stringibution_num',
                 'message_opp_ditf.stringibution_saturation', 'message_opp_limit_num', 'nationalEdu',
                 'number_of_dialogue', 'number_of_student_dialogue', 'online_opp_ditf.stringibution_num',
                 'online_opp_ditf.stringibution_saturation', 'online_opp_limit_num', 'opp_completed_rate',
                 'opp_following_num', 'opp_today_following_num', 'oppor_source', 'pastNday_advertiser_applied_ratio',
                 'pastNday_advertiser_ditf.stringibution_num', 'pastNday_advertiser_order_num',
                 'pastNday_applied_ratio_on_siteId', 'pastNday_ditf.stringibution_num_on_siteId',
                 'pastNday_order_num_on_siteId', 'past_n_days_acc_alignment_num', 'past_n_days_acc_order_amount',
                 'past_n_days_acc_order_avg_amount', 'past_n_days_acc_order_num', 'past_n_days_acc_order_rate',
                 'posLevel', 'promotion_type', 'quantum_id', 'race', 'recycle_opp_ditf.stringibution_num',
                 'recycle_opp_ditf.stringibution_saturation', 'recycle_opp_limit_num', 'residenceType', 'site_source',
                 'student_cellphone_midnum', 'student_dialogue_fenci', 'timedelta_from', 'today_unfollowed_num',
                 'twice_consult', 'valid_call_num']


def dump_category_values_2_file(csv_file, out_path):
    os.makedirs(out_path, exist_ok=True)

    feature_value_count_dict = {}
    for f in CATEGORY_FEATURES:
        feature_value_count_dict[f] = defaultdict(int)

    with open(csv_file, "r", "utf-8") as fin:
        for i, line in enumerate(fin):
            arr = line.strip().split(",")
            features = dict(zip(FEATURE_NAMES, arr))
            for f in CATEGORY_FEATURES:
                # 每种特征，统计其每种特征值的出现次数
                feature_value_count_dict[f][features[f]] += 1
            if i > 0 and i % 5000 == 0:
                print("line:", i)

    for f, f_dict in feature_value_count_dict.items():
        feature_values_file_out = os.path.join(category_value_path, f)
        with open(feature_values_file_out, 'w', 'utf-8') as fout:
            for k, v in sorted(f_dict.items(), key=lambda x: x[1], reverse=True):
                fout.write("{}\t{}\n".format(k, v))

    print("feature values dump completed!", category_value_path)


def read_feature_values_from_file(feature_name, feature_values_save_path=category_value_path):
    file_path = os.path.join(feature_values_save_path, feature_name)
    assert os.path.exists(file_path)
    value_count_dict = load_dict(file_path, splitor="\t", key_first=True)
    return list(value_count_dict.keys())


if __name__ == '__main__':
    from config.file_path_config import *

    dump_category_values_2_file(train_csv_file, category_value_path)

    create_user_set = read_feature_values_from_file("create_user", category_value_path)
    print(create_user_set)
