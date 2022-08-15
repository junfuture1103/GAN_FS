
import numpy as np
import pandas as pd

def sampling(df, size = 492):

    print("df_len : ", len(df))
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df  = df.loc[df['Class'] == 0][:size]

    normal_distributed_df= pd.concat([fraud_df, non_fraud_df])
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    print(new_df.head())

    print('Distribution of the Classes in the subsample dataset')
    
    print(new_df['Class'].value_counts())
    print(new_df['Class'].value_counts()/len(new_df))

    return new_df