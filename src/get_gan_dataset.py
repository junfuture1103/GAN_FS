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

def get_gan_datasets(gan):
    gan.fit()

    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    target_sample_num = total_neg_cnt - total_pos_cnt
    if target_sample_num <= 0:
        return full_dataset

    z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
    print("z in generate_samples", z)
    print("len z in generate_samples", len(z))
    new_samples = gan.generate_samples(z)
    print("new_samples len : ", len(new_samples))
    print("new_samples : ", new_samples)
    print("target_dataset.samples : ", target_dataset.samples)

    new_labels = torch.ones(target_sample_num, device=src.config.device)
    target_dataset.samples = torch.cat(
        [
            target_dataset.samples,
            new_samples,
        ],
    )
    target_dataset.labels = torch.cat(
        [
            target_dataset.labels,
            new_labels,
        ]
    )
    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    print("target_dataset.samples : ", len(target_dataset.samples))
    print("target_dataset.labels : ", len(target_dataset.labels))
    return target_dataset

def get_jgan_datasets(x, target_sample_num):
    # gan = src.gan.SNGAN()
    # gan = src.gan.WGANGP()
    # gan = src.gan.GAN()
    
    gan = src.gan.JUNGAN()
    gan.fit(x)

    target_dataset = src.datasets.FullDataset().to("cpu")

    start1 = time.time()
    # We do not user vae
    # z = vae.generate_z()
    z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
    # print("rand z in jun_gan : ", z)
    # print("rand z in jun_gan len : ", len(z))

    start2 = time.time()
    new_sample = gan.generate_samples(z)
    end2 = time.time()
    print("Generate_sample time : ", end2-start2)

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

    return target_dataset
