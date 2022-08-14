
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def box_graph(df, new_df):
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