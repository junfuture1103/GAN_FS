import numpy as np
import pandas as pd

from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src
# from sklearn.model_selection import train_test_split

def preprocess_data(data):
    print('=== STRAT PREPROCESS_DATA ===')

    # np_array = df.to_numpy()
    # print("np_array 0 : ", np_array[:,0])
    # np.random.shuffle(np_array)

    # partition labels and samples
    # labels = np_array[:, -1].copy()
    # samples = np_array[:,1:-2].copy()
    samples = data.loc[:, 'V1' : 'Amount']
    labels = data.loc[:, 'Class']
    # labels = np_array[:, -1].copy()
    # samples = np_array[:,:-1].copy()

    # print("samples in preprocess_data : ",samples)
    # new_label = []
    # for i in labels:
    #     print(i)
    #     if(i == ' positive'):
    #         new_label.append(1)
    #     else:
    #         new_label.append(0)
    # total data split to samples & labels complete
    # logger.info(f'total samples shape : {samples.shape}')
    # logger.info(f'total lables shape : {labels.shape}')

    # normalize samples -> min : 0 max : 1
    samples = minmax_scale(samples.astype('float32'))
    # labels = labels.astype('int')

    src.models.x_size = samples.shape[1]
    # total data spilt & samples normalization (0~1) complete
    # logger.info(f'samples normalization (0~1) complete')

    # return samples, np.array(new_label)
    return samples, labels

def prepare_dataset(name, training_test_ratio: float = 0.6) -> None:
    # new_df = anomaly_drop(name)
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        # random_state=src.config.seed,
    )
    src.datasets.training_samples = training_samples
    src.datasets.training_labels = training_labels
    src.datasets.test_samples = test_samples
    src.datasets.test_labels = test_labels

    return training_samples, test_samples, training_labels, test_labels

# def train_test_split(samples, labels, training_test_ratio: float = 0.1):
#     print('=== STRAT TRAIN_TEST_SPLIT ===')
#     #데이터 섞기
#     shuffle_index = np.random.permutation(len(samples))

#     x_data = samples[shuffle_index]
#     y_data = labels[shuffle_index]

#     # train data 갯수
#     n_train = int(len(x_data) * training_test_ratio)

#     x_train = x_data[:n_train]
#     y_train = y_data[:n_train]
#     x_test = x_data[n_train:]
#     y_test = y_data[n_train:]

#     # logger.info(f'total x_train shape : {x_train.shape}')
#     # logger.info(f'total y_train shape : {y_train.shape}')
#     # logger.info(f'total x_test shape : {x_test.shape}')
#     # logger.info(f'total y_test shape : {y_test.shape}')
#     # logger.info(f'traning_ratio : {training_test_ratio}, train_len : {len(x_train)}, test_len : {len(x_test)}')

#     return x_train, y_train, x_test, y_test

def anomaly_drop(new_df):
    # V14 특이치 제거하기 (가장 높은 Negative 상관 행렬을 가진 변수)
    v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
    v14_iqr = q75 - q25
    print('iqr: {}'.format(v14_iqr))

    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper  = q25 - v14_cut_off, q75 + v14_cut_off
    print('Cut off: {}'.format(v14_cut_off))
    print('V14 Lower: {}'.format(v14_lower))

    outliers = [x for x in v14_fraud if x < v14_lower or x> v14_upper]
    print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

    print('V14 outliers:{}'.format(outliers))

    new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
    print('----'*44)

    # V12 특이치 제거하기 
    V12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(V12_fraud, 25), np.percentile(V12_fraud, 75)
    print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
    V12_iqr = q75 - q25
    print('iqr: {}'.format(V12_iqr))

    V12_cut_off = V12_iqr * 1.5
    V12_lower, V12_upper  = q25 - V12_cut_off, q75 + V12_cut_off
    print('Cut off: {}'.format(V12_cut_off))
    print('V12 Lower: {}'.format(V12_lower))

    outliers = [x for x in V12_fraud if x < V12_lower or x> V12_upper]
    print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

    print('V12 outliers:{}'.format(outliers))

    new_df = new_df.drop(new_df[(new_df['V12'] > V12_upper) | (new_df['V12'] < V12_lower)].index)
    print('Number of Instances after outliers removal: {}'.format(len(new_df)))
    print('----'*44)

    # V10 특이치 제거하기 
    V10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(V10_fraud, 25), np.percentile(V10_fraud, 75)
    print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
    V10_iqr = q75 - q25
    print('iqr: {}'.format(V10_iqr))

    V10_cut_off = V10_iqr * 1.5
    V10_lower, V10_upper  = q25 - V10_cut_off, q75 + V10_cut_off
    print('Cut off: {}'.format(V10_cut_off))
    print('V10 Lower: {}'.format(V10_lower))

    outliers = [x for x in V10_fraud if x < V10_lower or x> V10_upper]
    print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

    print('V10 outliers:{}'.format(outliers))

    new_df = new_df.drop(new_df[(new_df['V10'] > V10_upper) | (new_df['V10'] < V10_lower)].index)
    print('Number of Instances after outliers removal: {}'.format(len(new_df)))
    print('----'*44)

    return new_df

def print_test_vs_train(train_labels, test_labels):
    print('=== TEST vs TRAIN ===')
    tmp = pd.DataFrame([[sum(train_labels == 0), sum(test_labels == 0), round(sum(train_labels == 0) / sum(train_labels == 1), 4)], [sum(train_labels == 1), sum(test_labels == 1), round(sum(test_labels == 0) / sum(test_labels == 1), 4)]], 
             columns=['train', 'test', 'percentage'], index=['0 (non-fraud)', '1 (fraud)'])
    print(tmp)


    return

def cnt_p_n(labels):
    pos_num = 0
    neg_num = 0

    pos_item_indices = []
    neg_item_indices = []

    idx = 0
    for label in labels:
        if(label == 1):
            pos_num += 1
            pos_item_indices.append(idx)
        else:
            neg_num += 1
            neg_item_indices.append(idx)
        idx += 1

    print("Major(neg) : ",neg_num)
    print("Minor(pos) : ",pos_num)

    return pos_item_indices, neg_item_indices