import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import minmax_scale
# from sklearn.model_selection import train_test_split

def preprocess_data(data):
    print('=== STRAT PREPROCESS_DATA ===')

    # get data complete
    # logger.info(f'get data complete data shape : {data.shape}')

    tmp = data['Class'].value_counts().to_frame().reset_index()
    tmp['Percent(%)'] = tmp["Class"].apply(lambda x : round(100*float(x) / len(data), 2))
    tmp = tmp.rename(columns = {"index" : "Target", "Class" : "Count"})
    
    samples = data.loc[:, 'V1' : 'Amount']
    labels = data.loc[:, 'Class']

    # total data split to samples & labels complete
    # logger.info(f'total samples shape : {samples.shape}')
    # logger.info(f'total lables shape : {labels.shape}')

    # normalize samples -> min : 0 max : 1
    samples = minmax_scale(samples.astype('float32'))
    labels = labels.astype('int')

    # total data spilt & samples normalization (0~1) complete
    # logger.info(f'samples normalization (0~1) complete')

    return samples, labels


def train_test_split(samples, labels, training_test_ratio: float = 0.1):
    print('=== STRAT TRAIN_TEST_SPLIT ===')
    #데이터 섞기
    shuffle_index = np.random.permutation(len(samples))

    x_data = samples[shuffle_index]
    y_data = labels[shuffle_index]

    # train data 갯수
    n_train = int(len(x_data) * training_test_ratio)

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test = x_data[n_train:]
    y_test = y_data[n_train:]

    # logger.info(f'total x_train shape : {x_train.shape}')
    # logger.info(f'total y_train shape : {y_train.shape}')
    # logger.info(f'total x_test shape : {x_test.shape}')
    # logger.info(f'total y_test shape : {y_test.shape}')
    # logger.info(f'traning_ratio : {training_test_ratio}, train_len : {len(x_train)}, test_len : {len(x_test)}')

    return x_train, y_train, x_test, y_test


def print_test_vs_train(train_labels, test_labels):
    print('=== TEST vs TRAIN ===')
    tmp = pd.DataFrame([[sum(train_labels == 0), sum(test_labels == 0)], [sum(train_labels == 1), sum(test_labels == 1)]], 
             columns=['train', 'test'], index=['0 (non-fraud)', '1 (fraud)'])
    print(tmp)

    return