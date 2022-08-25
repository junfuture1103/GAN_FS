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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import os
import src
from src import get_gan_dataset

FILE_NAME = 'creditcard.csv'
# FILE_NAME = 'wisconsin.dat'
# FILE_NAME = 'test_result.csv'

if __name__ == '__main__':
    # read data
    print(os.path.realpath(__file__))
    df =pd.read_csv('./creditcard.csv')
    df.columns

    # src.utils.set_random_state()
    # src.utils.prepare_dataset(FILE_NAME)

    # #random sampling
    df = df.sample(frac=1)
    # #get data -> balancing pos == neg 기준 : pos갯수
    new_df = src.sample.sampling(df, 5000)
    # print(new_df.columns)
    # # data process
    # train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(df)
    # for fast test
    # samples, labels = src.process_datasets.preprocess_data(new_df)
    train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(new_df)

    # test & train split complete
    # src.process_datasets.print_test_vs_train(train_labels, test_labels)
    # train_pos_idx, train_neg_idx = src.process_datasets.cnt_p_n(src.datasets.training_labels)  
    train_pos_idx, train_neg_idx = src.process_datasets.cnt_p_n(train_labels)   

    # get positive datasets & negative datasets
    train_pos_dataset = src.datasets.training_samples[train_pos_idx]
    train_neg_dataset = src.datasets.training_samples[train_neg_idx]
    
    print("train_pos_dataset type : ", type(train_pos_dataset))
    print("train_neg_dataset type : ", type(train_neg_dataset))

    total_pos_cnt = len(train_pos_dataset)
    total_neg_cnt = len(train_neg_dataset)

    print("total_post_cnt : ", total_pos_cnt)
    print("total_neg_cnt : ", total_neg_cnt)

    target_sample_num = total_neg_cnt - total_pos_cnt

    x = torch.from_numpy(train_pos_dataset).to("cpu")
    gan_dataset = get_gan_dataset.get_jgan_datasets(x, target_sample_num)

    # x = torch.from_numpy(train_pos_dataset).to("cpu")
    # gan_dataset2 = get_gan_dataset.get_jgan_datasets(x, target_sample_num)
    # 샘플링
    # SMOTE 객체 생성
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(src.datasets.training_samples, src.datasets.training_labels)
    
    train_pos_idx, train_neg_idx = src.process_datasets.cnt_p_n(y_train_resampled)    

    # get positive datasets & negative datasets
    train_pos_dataset = X_train_resampled[train_pos_idx]
    train_neg_dataset = X_train_resampled[train_neg_idx]
    
    # print("train_pos_dataset type : ", type(train_pos_dataset))
    # print("train_neg_dataset type : ", type(train_neg_dataset))

    total_pos_cnt = len(train_pos_dataset)
    total_neg_cnt = len(train_neg_dataset)

    print("total_post_cnt in SMOTE : ", total_pos_cnt)
    print("total_neg_cnt in SMOTE : ", total_neg_cnt)

    target_sample_num = total_neg_cnt - total_pos_cnt

    # x = torch.from_numpy(train_pos_dataset).to("cpu")
    # gan_dataset2 = get_gan_dataset.get_jgan_datasets(x, target_sample_num)

    print("len resampled X : ", len(X_train_resampled))
    print("len resampled Y : ", len(y_train_resampled))

    # # autoencoding
    # src.autoencoder.ae(train_samples, train_labels, test_samples, test_labels)

    # # x -> samples, y -> labels
    # start1 = time.time()
    # print("============ Start t-SNE for Visualization ============")
    
    # # src.plot.TSNE_graph(X_train_resampled, y_train_resampled, gan_dataset2.samples, gan_dataset2.labels)
    # # src.plot.TSNE_graph(gan_dataset.samples, gan_dataset.labels, train_samples, train_labels)
    # end1 = time.time()
    # print("t-SNE time : ", end1-start1)

    # print("============ vs_traditional_methods ============")
    # RandomOverSampler,SMOTE, ADASYN, BorderlineSMOTE
    src.assessment_metric.vs_traditional_methods(train_samples, test_samples, train_labels, test_labels)

    # src.classifier.Classifier(gan_dataset)
    print("============ Just RF ============")
    src.regression.RandomForest(train_samples, train_labels, test_samples, test_labels)
    print("============ RF with GAN ============")
    src.regression.RandomForest(gan_dataset.samples, gan_dataset.labels, test_samples, test_labels)
    print("============ Just LGBM ============")
    src.regression.LGBM(train_samples, train_labels, test_samples, test_labels)
    print("============ LGBM with GAN ============")
    src.regression.LGBM(gan_dataset.samples, gan_dataset.labels, test_samples, test_labels)
    print("============ LGBM with SMOTE ============")
    src.regression.LGBM(X_train_resampled, y_train_resampled, test_samples, test_labels)