# -*- coding: utf-8 -*-

"""
@author: zhangmeng
@file: deep_feature_config.py
@time: 2019/12/10 17:17
"""
import tensorflow as tf

FEATURE_INFOS = [

    # [is_use, f_name, f_type, default, norm_method]

    [1, "label", tf.string, "", None],

    [1, "account_age", tf.int64, 0, "min_max"],
    [1, "alignment_day", tf.string, "", None],
    [1, "alignment_day_of_week", tf.string, "", None],
    [1, "alignment_hour", tf.string, "", None],
    [1, "alignment_month", tf.string, "", None],
    [1, "alignment_year", tf.string, "", None],
    [1, "all_call_num", tf.float32, 0.0, "min_max"],
    [1, "avgerage_waiting_time_of_dialogue", tf.float32, 0.0, "min_max"],
    [1, "consult_type", tf.string, "", None],
    [1, "contain_education_or_promotion_key_word_in_dialogue", tf.string, "", None],
    [1, "contain_price_key_word_in_dialogue", tf.string, "", None],
    [1, "create_user", tf.string, "", None],
    [1, "delta_days_from_entryDate", tf.int64, 0, "min_max"],
    [1, "delta_time_of_create_time_and_operator_time", tf.float32, 0.0, "min_max"],
    [1, "dialogue_start_at", tf.string, "", None],
    [1, "duration_of_dialogue", tf.float32, 0.0, "min_max"],
    [1, "educationalBackground", tf.string, "", None],
    [1, "first_proj_id", tf.string, "", None],
    [1, "gender", tf.string, "", None],
    [1, "graduateSchool", tf.string, "", None],
    [1, "major", tf.string, "", None],
    [1, "marriageState", tf.string, "", None],
    [1, "message_opp_distribution_num", tf.int64, 0, "min_max"],
    [1, "message_opp_distribution_saturation", tf.float32, 0.0, "min_max"],
    [1, "message_opp_limit_num", tf.int64, 0, "min_max"],
    [1, "nationalEdu", tf.string, "", None],
    [1, "number_of_dialogue", tf.int64, 0, "min_max"],
    [1, "number_of_student_dialogue", tf.int64, 0, "min_max"],
    [1, "online_opp_distribution_num", tf.int64, 0, "min_max"],
    [1, "online_opp_distribution_saturation", tf.float32, 0.0, "min_max"],
    [1, "online_opp_limit_num", tf.int64, 0, "min_max"],
    [1, "opp_completed_rate", tf.float32, 0.0, "min_max"],
    [1, "opp_following_num", tf.int64, 0, "min_max"],
    [1, "opp_today_following_num", tf.int64, 0, "min_max"],
    [1, "oppor_source", tf.string, "", None],
    [1, "pastNday_advertiser_applied_ratio", tf.float32, 0.0, "min_max"],
    [1, "pastNday_advertiser_distribution_num", tf.int64, 0, "min_max"],
    [1, "pastNday_advertiser_order_num", tf.int64, 0, "min_max"],
    [1, "pastNday_applied_ratio_on_siteId", tf.float32, 0.0, "min_max"],
    [1, "pastNday_distribution_num_on_siteId", tf.int64, 0, "min_max"],
    [1, "pastNday_order_num_on_siteId", tf.int64, 0, "min_max"],
    [1, "past_n_days_acc_alignment_num", tf.int64, 0, "min_max"],
    [1, "past_n_days_acc_order_amount", tf.float32, 0.0, "min_max"],
    [1, "past_n_days_acc_order_avg_amount", tf.float32, 0.0, "min_max"],
    [1, "past_n_days_acc_order_num", tf.int64, 0, "min_max"],
    [1, "past_n_days_acc_order_rate", tf.float32, 0.0, "min_max"],
    [1, "posLevel", tf.string, "", None],
    [1, "promotion_type", tf.string, "", None],
    [1, "quantum_id", tf.string, "", None],
    [1, "race", tf.string, "", None],
    [1, "recycle_opp_distribution_num", tf.int64, 0, "min_max"],
    [1, "recycle_opp_distribution_saturation", tf.float32, 0.0, "min_max"],
    [1, "recycle_opp_limit_num", tf.int64, 0, "min_max"],
    [1, "residenceType", tf.string, "", None],
    [1, "site_source", tf.string, "", None],
    [1, "student_cellphone_midnum", tf.string, "", None],
    [0, "student_dialogue_fenci", tf.string, "", None],
    [1, "timedelta_from", tf.float32, 0.0, "min_max"],
    [1, "today_unfollowed_num", tf.int64, 0, "min_max"],
    [1, "twice_consult", tf.string, "", None],
    [1, "valid_call_num", tf.int64, 0, "min_max"],
]


def parse_feature_config():
    feature_name_list = []
    feature_dtype_dict = {}
    feature_default_dict = {}
    feature_use_list = []
    feature_name_use_list = []
    category_feature_list = []
    continuous_feature_list = []
    min_max_norm_method_ls = []
    log_min_max_norm_method_ls = []

    feature_keras_input_dict = {}
    for i, arr in enumerate(FEATURE_INFOS):
        use, f_name, dtype, default, norm_method = arr
        feature_name_list.append(f_name)
        feature_dtype_dict[f_name] = dtype

        assert dtype in [tf.string, tf.int64, tf.float32]

        if use == 1:
            feature_use_list.append(i)
            feature_name_use_list.append(f_name)
            feature_default_dict[f_name] = default

            if f_name != "label":

                feature_keras_input_dict[f_name] = tf.keras.Input(name=f_name, shape=(1,), dtype=dtype)

                if dtype == tf.string:
                    category_feature_list.append(f_name)
                else:
                    continuous_feature_list.append(f_name)

                # 连续值处理方式分组
                if norm_method == "min_max":
                    min_max_norm_method_ls.append(f_name)
                if norm_method == "log_min_max":
                    log_min_max_norm_method_ls.append(f_name)

    return feature_name_list, feature_dtype_dict, feature_default_dict, feature_use_list, feature_name_use_list, \
           category_feature_list, continuous_feature_list, feature_keras_input_dict, min_max_norm_method_ls, log_min_max_norm_method_ls


FEATURE_NAMES, FEATURE_DTYPE_DICT, \
FEATURE_DEFAULT_DICT, FEATURE_USE_LIST, FEATURE_NAME_USE_LIST, \
CATEGORY_FEATURES, CONTINUOUS_FEATURES, \
FEATURE_KERAS_INPUT_DICT, \
MIN_MAX_METHOD_LIST, LOG_MIN_MAX_METHOD_LIST = parse_feature_config()

# 参数初步校验
assert len(FEATURE_NAMES) == len(FEATURE_DTYPE_DICT)
assert len(FEATURE_USE_LIST) == len(FEATURE_DEFAULT_DICT) == len(FEATURE_NAME_USE_LIST)
assert len(CATEGORY_FEATURES) + len(CONTINUOUS_FEATURES) == len(FEATURE_KERAS_INPUT_DICT)
assert len(FEATURE_NAME_USE_LIST) == len(FEATURE_KERAS_INPUT_DICT) + 1


# 从csv文件读取dataset

def _parse_line(line, csv_feature_defaults):
    fields = tf.io.decode_csv(line, csv_feature_defaults, select_cols=FEATURE_USE_LIST)
    features = dict(zip(FEATURE_NAME_USE_LIST, fields))
    label = features.pop("label")
    return features, label


def read_csv_2_dataset(csv_file, mean_dict_save_path, shuffle_size=10000, batch_size=256):
    # 加载csv默认值list
    mean_dict = load_mean_dict(mean_dict_save_path)
    csv_feature_defaults = [
        mean_dict[f] if f in mean_dict else FEATURE_DEFAULT_DICT[f] for f in FEATURE_NAME_USE_LIST]
    # print(csv_feature_defaults)

    data = tf.data.TextLineDataset(csv_file)
    data = data.shuffle(shuffle_size).map(lambda x: _parse_line(x, csv_feature_defaults)).batch(batch_size)
    return data


def load_mean_dict(save_path):
    import os
    import pickle
    assert os.path.exists(save_path)
    save_dict = pickle.load(open(save_path, "rb"))
    mean_dict = {}
    for key, (min_value, max_value, mean_value, _, _) in save_dict.items():
        mean_dict[key] = mean_value
    return mean_dict


if __name__ == '__main__':
    # for i in [FEATURE_NAMES, FEATURE_DTYPES, FEATURE_DEFAULTS, FEATURE_KERAS_INPUT_DICT]:
    #     print(i)

    print(CATEGORY_FEATURES)
    print(CONTINUOUS_FEATURES)
    print(FEATURE_NAMES)
    print(MIN_MAX_METHOD_LIST)
    print(LOG_MIN_MAX_METHOD_LIST)

    from config.file_path_config import test_csv_file, min_max_value_path

    dataset = read_csv_2_dataset(test_csv_file, min_max_value_path, shuffle_size=10000, batch_size=256)
    cnt = 0
    for d in dataset:
        cnt += 1
        print(d[1])
        for k, v in d[0].items():
            print(k)
            print(v)
        if cnt > 1:
            break
