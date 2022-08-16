
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE

F, (ax1, ax2) = plt.subplots(2,1, figsize=(24,20))
colors = ["#0101DF", "#DF0101"]

def corrlation(df, new_df):
    # Imbalanced Corr
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
    ax1.set_title("Imbalanced Colrrelation Matrix \n (don't use for reference)", fontsize=14)

    # balanced Corr
    sub_sample_corr = new_df.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20},ax=ax2 )
    ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
    
    # show Correlation Graph
    plt.show()

def box_graph(new_df):
    f, axes = plt.subplots(ncols=4, figsize = (20,4))
    sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
    axes[0].set_title('V17 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
    axes[1].set_title('V14 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
    axes[2].set_title('V12 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
    axes[3].set_title('V10 vs Class Negative Correlation')

    plt.show()

def TSNE_graph(target_sample, target_label, train_sample, train_label):

    X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(target_sample)
    X_reduced_tsne2 = TSNE(n_components=2, random_state=42).fit_transform(train_sample)

    f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(24,6))
    f.suptitle('Clustering using Dimensionality Reduction', fontsize = 14)

    blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
    red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
    color_num = 2

    y = target_label
    # t-SNE scatter plot
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y==0), cmap= plt.cm.coolwarm, label='No Fraud', linewidths=2)
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y==1), cmap= plt.cm.coolwarm, label='Fraud', linewidths=2)
    ax1.set_title('t-SNE', fontsize=14)

    ax1.grid(True)
    ax1.legend(handles =[blue_patch, red_patch])

    y = train_label
    # t-SNE scatter plot
    ax2.scatter(X_reduced_tsne2[:,0], X_reduced_tsne2[:,1], c=(y==0), cmap= plt.cm.coolwarm, label='No Fraud', linewidths=2)
    ax2.scatter(X_reduced_tsne2[:,0], X_reduced_tsne2[:,1], c=(y==1), cmap= plt.cm.coolwarm, label='Fraud', linewidths=2)
    ax2.set_title('t-SNE', fontsize=14)

    ax2.grid(True)
    ax2.legend(handles =[blue_patch, red_patch])

    plt.show()