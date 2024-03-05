import numpy as np
import pandas as pd


def loadData(windows, num_samples, down):
    data = pd.read_csv('./Data/measures_v2.csv')
    input_cols = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q', 'i_s', 'u_s']
    target_cols = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
    temperature_cols = target_cols + ['ambient', 'coolant']
    val_profiles = [58]
    test_profiles = [65, 72]
    # test_profiles = [65]
    train_profiles = [p for p in data.profile_id.unique() if p not in val_profiles + test_profiles]
    new_cols = ['profile_id'] + input_cols + target_cols

    temperature_scale = 100  # in deg C
    non_temperature_cols = [c for c in data if c not in temperature_cols + ['profile_id']]
    data.loc[:, temperature_cols] /= temperature_scale
    data.loc[:, non_temperature_cols] /= data.loc[:, non_temperature_cols].abs().max(axis=0)

    # extra feats (FE)
    if {'i_d', 'i_q', 'u_d', 'u_q'}.issubset(set(data.columns.tolist())):
        extra_feats = {'i_s': lambda x: np.sqrt((x['i_d'] ** 2 + x['i_q'] ** 2)),
                       'u_s': lambda x: np.sqrt((x['u_d'] ** 2 + x['u_q'] ** 2))}
    data = data.assign(**extra_feats)
    data = data[new_cols]
    train = data[data['profile_id'].isin(train_profiles)]
    valid = data[data['profile_id'].isin(val_profiles)]
    test = data[data['profile_id'].isin(test_profiles)]
    train = arr2seq(train, windows, down)
    valid = arr2seq_test(valid, windows, down)
    test = arr2seq_test(test, windows, down)
    if num_samples < train.shape[0]:
        random_rows = np.random.choice(train.shape[0], size=int(num_samples), replace=False)
        train = train[random_rows]
    return train, valid, test


def MeanDown(array, down):
    n = array.shape[0]
    merge = []
    i = 0
    while n - i > down:
        temp = np.mean(array[i:i + down, :], axis=0)
        merge.append(temp)
        i += down
    if i != n - 1:
        merge.append(np.mean(array[i:, :], axis=0))
    return np.array(merge)


def arr2seq(data, seqlen, down):
    prediction_length, estimation_length = seqlen
    profiles = [p for p in data.profile_id.unique()]
    arr = []
    for index in profiles:
        temp = data[data['profile_id'] == index].to_numpy()
        down_samples = MeanDown(temp, down)
        arr.append(down_samples)
    ret = []
    for profile in arr:
        if profile.shape[0] > prediction_length+estimation_length:
            for i in range(profile.shape[0] - (prediction_length+estimation_length)):
                ret.append(profile[i:i + (prediction_length+estimation_length), 1:])
        else:
            continue
    return np.array(ret)


def arr2seq_test(data, seqlen, down):
    prediction_length, estimation_length = seqlen
    profiles = [p for p in data.profile_id.unique()]
    arr = []
    for index in profiles:
        temp = data[data['profile_id'] == index].to_numpy()
        down_samples = MeanDown(temp, down)
        arr.append(down_samples)
    ret = []
    for profile in arr:
        if profile.shape[0] > (prediction_length+estimation_length):
            i = 0
            while True:
                if i + (prediction_length+estimation_length) > profile.shape[0]:
                    break
                ret.append(profile[i:i + (prediction_length+estimation_length), 1:])
                i += estimation_length-1
        else:
            continue
    return np.array(ret)
