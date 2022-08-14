import random
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf

import src

FILE_NAME = 'creditcard.csv'
# FILE_NAME = 'result2.csv'
# FILE_NAME = 'test_result.csv'

if __name__ == '__main__':
    # read data
    data = pd.read_csv(FILE_NAME)
    data.shape

    # data process
    total_samples, total_labels = src.datasets.preprocess_data(data)
    train_samples, train_labels, test_samples, test_labels = src.datasets.train_test_split(total_samples, total_labels)
    
    # test & train split complete
    src.datasets.print_test_vs_train(train_labels, test_labels)

    #random sampling
    df = data.sample(frac=1)
    print(len(df))
    new_df = src.sample.sampling(df)

    # plot
    # src.plot.corrlation(df,new_df)
    # src.plot.box_graph(df,new_df)

    # autoencoding
    src.autoencoder.ae(train_samples, train_labels, test_samples, test_labels)

    # x -> samples, y -> labels
    src.regression.RandomForest(train_samples, train_labels, test_samples, test_labels)