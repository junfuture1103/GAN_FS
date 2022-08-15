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
    # data.shape

    # data process
    train_samples, test_samples, train_labels, test_labels = src.procss_datasets.prepare_dataset(data)
    # print(train_samples, test_samples, train_labels, test_labels)


    # test & train split complete
    src.procss_datasets.print_test_vs_train(train_labels, test_labels)
    train_pos_idx, train_neg_idx = src.procss_datasets.cnt_p_n(train_labels)    

    # get positive datasets & negative datasets
    train_pos_dataset = train_samples[train_pos_idx]
    train_neg_dataset = train_samples[train_neg_idx]
    
    #random sampling
    df = data.sample(frac=1)
    # print(len(df))
    #get data -> balancing pos == neg 기준 : pos갯수
    new_df = src.sample.sampling(df)

    # # plot
    # src.plot.corrlation(df,new_df)
    # src.plot.box_graph(df,new_df)

    x = torch.from_numpy(train_samples[train_pos_idx]).to("cpu")

    vae = src.vae.VAE()
    vae.fit()
    gan = src.gan.GAN()
    gan.fit(x)

    target_dataset = src.datasets.FullDataset().to("cpu")
    total_pos_cnt = len(train_pos_dataset)
    total_neg_cnt = len(train_neg_dataset)

    while True:
        print("total_post_cnt : ", total_pos_cnt)
        print("total_neg_cnt : ", total_neg_cnt)

        if total_pos_cnt >= total_neg_cnt:
            break
        else:
            # update the number of positive samples
            z = vae.generate_z()
            new_sample = gan.generate_samples(z)
            new_label = torch.tensor([1], device="cpu")

            target_dataset.samples = torch.cat(
                [
                    target_dataset.samples,
                    new_sample,
                ],
            )
            target_dataset.labels = torch.cat(
                [
                    target_dataset.labels,
                    new_label,
                ]
            )
            total_pos_cnt += 1

            # # update the number of overlapping positive samples
            # indices = get_knn_indices(new_sample, full_dataset.samples)
            # labels = full_dataset.labels[indices]

            # if 0 in labels:
            #     print("ol_pos_cnt int while : ", ol_pos_cnt)
            #     ol_pos_cnt += 1

    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    print("target_dataset.samples : ", len(target_dataset.samples))
    print("target_dataset.labels : ", len(target_dataset.labels))

    # # autoencoding
    # src.autoencoder.ae(train_samples, train_labels, test_samples, test_labels)

    # # x -> samples, y -> labels
    # src.regression.RandomForest(train_samples, train_labels, test_samples, test_labels)