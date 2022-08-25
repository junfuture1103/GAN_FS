import torch
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
import random
import src
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

DATASET = 'creditcard.csv'
# DATASET = 'ecoli3.dat'
# DATASET = 'wisconsin.dat'

TRADITIONAL_METHODS = [
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
]

GAN_MODELS = [
    src.gan.GAN,
    src.gan.WGAN,
    src.gan.WGANGP,
    src.gan.SNGAN,
    src.gan.JUNGAN,
    # src.gans.RVGAN,
    # src.gans.RVWGAN,
    # src.gans.RVWGANGP,
    # src.gans.RVSNGAN,
]

def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    result = dict()
    # read data
    data = pd.read_csv(DATASET)
    df = data.sample(frac=1)

    new_df = src.sample.sampling(df, 600)
    
    # train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(df)
    train_samples, test_samples, train_labels, test_labels = src.process_datasets.prepare_dataset(new_df)

    raw_x, raw_y = train_samples, train_labels
    print("raw_x,", raw_x)
    print("raw_y,", raw_y)
    # raw_x = raw_x.numpy()
    # raw_y = raw_y.numpy()

    for M in TRADITIONAL_METHODS:
        x, _ = M(random_state=src.config.seed).fit_resample(raw_x, raw_y)
        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            # learning_rate='auto',
            init='random',
            n_components=2,
            random_state=src.config.seed,
        ).fit_transform(x)
        result[M.__name__] = [embedded_x, y]

    train_pos_idx, train_neg_idx = src.process_datasets.cnt_p_n(train_labels)    
    
    # get positive datasets & negative datasets
    train_pos_dataset = train_samples[train_pos_idx]
    train_neg_dataset = train_samples[train_neg_idx]
    
    print("train_labels in tsne_all : ", train_labels[train_pos_idx])
    total_pos_cnt = len(train_pos_dataset)
    total_neg_cnt = len(train_neg_dataset)

    target_sample_num = total_neg_cnt - total_pos_cnt

    print(train_pos_dataset[-1])

    for M in GAN_MODELS:
        x = torch.from_numpy(train_pos_dataset).to("cpu")
        set_random_state()
        gan = M()
        gan.fit(x)
        # z = torch.randn([len(raw_y) - int(2 * sum(raw_y)), src.models.z_size], device=src.config.device)
        z = torch.randn([target_sample_num, src.models.z_size], device=src.config.device)
        x = np.concatenate([raw_x, gan.g(z).detach().cpu().numpy()])

        # print(gan.g(z).detach().cpu().numpy())

        y = np.concatenate([raw_y, np.full(len(x) - len(raw_x), 2)])
        embedded_x = TSNE(
            # learning_rate='auto',
            init='random',
            random_state=src.config.seed,
        ).fit_transform(x)

        result[M.__name__] = [embedded_x, y]

    sns.set_style('white')
    fig, axes = plt.subplots(2, 4)
    for (key, value), axe in zip(result.items(), axes.flat):
        # axe.set(xticklabels=[])
        # axe.set(yticklabels=[])
        axe.set(title=key)
        majority = []
        minority = []
        generated_data = []
        for i, j in zip(value[0], value[1]):
            if j == 0:
                majority.append(i)
            elif j == 1:
                minority.append(i)
            else:
                generated_data.append(i)
        minority = np.array(minority)
        majority = np.array(majority)
        generated_data = np.array(generated_data)
        sns.scatterplot(
            x=majority[:, 0],
            y=majority[:, 1],
            ax=axe,
            alpha=0.5,
            label='majority',
        )
        sns.scatterplot(
            x=generated_data[:, 0],
            y=generated_data[:, 1],
            ax=axe,
            alpha=0.5,
            label='generated_data',
        )
        sns.scatterplot(
            x=minority[:, 0],
            y=minority[:, 1],
            ax=axe,
            alpha=1.0,
            s=10,
            label='minority',
        )
        axe.get_legend().remove()

    fig.set_size_inches(18, 10)
    fig.set_dpi(100)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    plt.savefig('all_distribution.jpg')
    plt.show()
