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

from sklearn.manifold import TSNE
import src

FILE_NAME = 'creditcard.csv'
# FILE_NAME = 'result2.csv'
# FILE_NAME = 'test_result.csv'

if __name__ == '__main__':
    # read data
    data = pd.read_csv(FILE_NAME)
    df = data.sample(frac=1)

    new_df = src.sample.sampling(df, 1000)

    # data process
    train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(df)
    train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(new_df)

    # test & train split complete
    src.process_datasets.print_test_vs_train(train_labels, test_labels)
    train_pos_idx, train_neg_idx = src.process_datasets.cnt_p_n(train_labels)    

    # get positive datasets & negative datasets
    train_pos_dataset = train_samples[train_pos_idx]
    train_neg_dataset = train_samples[train_neg_idx]
    
    print("train_pos_dataset type : ", type(train_pos_dataset))
    print("train_neg_dataset type : ", type(train_neg_dataset))

    # # plot
    # src.plot.corrlation(df,new_df)
    # src.plot.box_graph(df)

    x = torch.from_numpy(train_samples[train_pos_idx]).to("cpu")

    gan = src.gan.GAN()
    gan.fit(x)

    target_dataset = src.datasets.FullDataset().to("cpu")
    total_pos_cnt = len(train_pos_dataset)
    total_neg_cnt = len(train_neg_dataset)

    print("total_post_cnt : ", total_pos_cnt)
    print("total_neg_cnt : ", total_neg_cnt)
    
    target_sample_num = total_neg_cnt - total_pos_cnt

    start = time.time()
    z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
    end = time.time()
    print("Generate_z time : ", end-start)

    start = time.time()
    new_sample = gan.generate_samples(z)
    end = time.time()
    print("Generate_sample time : ", end-start)

    new_label = torch.ones(target_sample_num, device=src.config.device)

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

    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    print("target_dataset.samples : ", len(target_dataset.samples))
    print("target_dataset.labels : ", len(target_dataset.labels))


    # # x -> samples, y -> labels
    start = time.time()
    print("============ Start t-SNE for Visualization ============")
    src.plot.TSNE_graph(target_dataset.samples, target_dataset.labels, train_samples, train_labels)
    end = time.time()

    print("t-SNE time : ", end-start)