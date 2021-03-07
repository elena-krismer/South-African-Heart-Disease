import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd


def preprocessing(df):
    raw_data = df.values
    X = raw_data[:, 1:-1]
    # preprocessing attribute family history
    X[X[:, 4] == 'Absent', 4] = 0
    X[X[:, 4] == 'Present', 4] = 1
    # from object array to float array(else SVD raises error)
    X = np.array(X, dtype=np.float64)
    y = raw_data[:, -1]
    N, M = X.shape
    attributeNames = np.asarray(df.columns[1:-1])
    return X, M, attributeNames


def printStatistics(X, M, attributeNames):
    for i in range(0, M):
        print('\nattribute:', attributeNames[i])
        print('Mean:', X[:, i].mean())
        print('Standard Deviation:', X[:, i].std(ddof=1))
        print('Median:', np.median(X[:, i]))
        print('Range:', X[:, i].max() - X[:, i].min())


filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)
nonCHD_df = df[df.chd == 0]
CHD_df = df[df.chd == 1]

nonCHD_df_data = preprocessing(nonCHD_df)
CHD_df_data = preprocessing(nonCHD_df)
printStatistics(nonCHD_df_data[0], nonCHD_df_data[1], nonCHD_df_data[2])
printStatistics(CHD_df_data[0], CHD_df_data[1], CHD_df_data[2])
