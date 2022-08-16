import numpy as np
import pandas as pd

from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src
# from sklearn.model_selection import train_test_split

def preprocess_data(df):
    print('=== STRAT PREPROCESS_DATA ===')

    np_array = df.to_numpy()
    print("np_array 0 : ", np_array[:,0])
    np.random.shuffle(np_array)

    # partition labels and samples
    labels = np_array[:, -1].copy()
    samples = np_array[:,:-1].copy()

    # total data split to samples & labels complete
    # logger.info(f'total samples shape : {samples.shape}')
    # logger.info(f'total lables shape : {labels.shape}')

    # normalize samples -> min : 0 max : 1
    samples = minmax_scale(samples.astype('float32'))
    labels = labels.astype('int')

    src.models.x_size = samples.shape[1]
    # total data spilt & samples normalization (0~1) complete
    # logger.info(f'samples normalization (0~1) complete')

    return samples, labels

def prepare_dataset(name: str, training_test_ratio: float = 0.6) -> None:
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