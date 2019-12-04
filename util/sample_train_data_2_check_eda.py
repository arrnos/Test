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

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
np.set_printoptions(threshold=np.inf)
sns.set(style="whitegrid")

cate_features =['alignment_day', 'alignment_day_of_week', 'alignment_hour', 'alignment_month', 'alignment_year', 'consult_type', 'contain_education_or_promotion_key_word_in_dialogue', 'contain_price_key_word_in_dialogue', 'create_user', 'dialogue_start_at', 'educationalBackground', 'first_proj_id', 'gender', 'graduateSchool', 'major', 'marriageState', 'nationalEdu', 'oppor_source', 'posLevel', 'promotion_type', 'quantum_id', 'race', 'residenceType', 'site_source', 'student_cellphone_midnum', 'twice_consult']
cont_features = ['account_age', 'all_call_num', 'avgerage_waiting_time_of_dialogue', 'delta_days_from_entryDate', 'delta_time_of_create_time_and_operator_time', 'duration_of_dialogue', 'message_opp_distribution_num', 'message_opp_distribution_saturation', 'message_opp_limit_num', 'number_of_dialogue', 'number_of_student_dialogue', 'online_opp_distribution_num', 'online_opp_distribution_saturation', 'online_opp_limit_num', 'opp_completed_rate', 'opp_following_num', 'opp_today_following_num', 'pastNday_advertiser_applied_ratio', 'pastNday_advertiser_distribution_num', 'pastNday_advertiser_order_num', 'pastNday_applied_ratio_on_siteId', 'pastNday_distribution_num_on_siteId', 'pastNday_order_num_on_siteId', 'past_n_days_acc_alignment_num', 'past_n_days_acc_order_amount', 'past_n_days_acc_order_avg_amount', 'past_n_days_acc_order_num', 'past_n_days_acc_order_rate', 'recycle_opp_distribution_num', 'recycle_opp_distribution_saturation', 'recycle_opp_limit_num', 'timedelta_from', 'today_unfollowed_num', 'valid_call_num']
csv_features = ['label', 'account_age', 'alignment_day', 'alignment_day_of_week', 'alignment_hour', 'alignment_month', 'alignment_year', 'all_call_num', 'avgerage_waiting_time_of_dialogue', 'consult_type', 'contain_education_or_promotion_key_word_in_dialogue', 'contain_price_key_word_in_dialogue', 'create_user', 'delta_days_from_entryDate', 'delta_time_of_create_time_and_operator_time', 'dialogue_start_at', 'duration_of_dialogue', 'educationalBackground', 'first_proj_id', 'gender', 'graduateSchool', 'major', 'marriageState', 'message_opp_distribution_num', 'message_opp_distribution_saturation', 'message_opp_limit_num', 'nationalEdu', 'number_of_dialogue', 'number_of_student_dialogue', 'online_opp_distribution_num', 'online_opp_distribution_saturation', 'online_opp_limit_num', 'opp_completed_rate', 'opp_following_num', 'opp_today_following_num', 'oppor_source', 'pastNday_advertiser_applied_ratio', 'pastNday_advertiser_distribution_num', 'pastNday_advertiser_order_num', 'pastNday_applied_ratio_on_siteId', 'pastNday_distribution_num_on_siteId', 'pastNday_order_num_on_siteId', 'past_n_days_acc_alignment_num', 'past_n_days_acc_order_amount', 'past_n_days_acc_order_avg_amount', 'past_n_days_acc_order_num', 'past_n_days_acc_order_rate', 'posLevel', 'promotion_type', 'quantum_id', 'race', 'recycle_opp_distribution_num', 'recycle_opp_distribution_saturation', 'recycle_opp_limit_num', 'residenceType', 'site_source', 'student_cellphone_midnum', 'student_dialogue_fenci', 'timedelta_from', 'today_unfollowed_num', 'twice_consult', 'valid_call_num']

csv_file = "E:\project_data\Test\\train_csv_feature_sample"

def load_data_2_df():
    df = pd.read_csv(csv_file)
    df.columns = csv_features

    feature_ls  = list([(x,'category') for x in ["label"]+cate_features]+[(x,float) for x in cont_features])
    dtypes = dict(feature_ls)
    df = df.astype(dtypes)
    return df

def eda_for_cate_features(df):
    n = len(df)
    for feature in cate_features:
        feature_value = df[feature]
        print("\n[%s]"%feature)
        print("缺失率：%.0f%%"%(feature_value.isna().sum()/n*100))
        value_counts = feature_value.value_counts()
        print("value_counts:{}".format(len(value_counts)))
        print(value_counts[:50])
        if len(value_counts)<20:
            sns.countplot(x=feature_value)
            plt.show()
        else:
            sns.violinplot(x = value_counts.values)
            plt.show()
