import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd


# Scatter plot 2 attributes against each other
def plot2attributes(att1, att2, X, y, attributeNames, classNames, C):
    plt.figure()
    plt.title('SA heart disease data')
    for c in range(C):
        class_mask = y == c
        plt.plot(X[class_mask, att1], X[class_mask, att2], 'o', alpha=.3)
    plt.legend(classNames)
    plt.xlabel(attributeNames[att1])
    plt.ylabel(attributeNames[att2])
    plt.show()


# Plot component variance
def plotComponentVariance(X):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    Y=Y*(1/np.std(Y,0))

    U, S, V = svd(Y, full_matrices=False)
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


# Scatter plot 2 components
def plot2components(X):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    Y=Y*(1/np.std(Y,0))

    U, S, V = svd(Y, full_matrices=False)

    V = V.T
    Z = Y @ V

    i = 0
    j = 1

    # Plot PCA of the data
    plt.figure()
    plt.title('NanoNose data: PCA')
    # Z = array(Z)
    for c in range(C):
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))
    plt.show()

def plotHist(X, N, M, attributeNames):
    plt.figure(figsize=(12, 12))
    u = np.floor(np.sqrt(M))
    v = np.ceil(float(M) / u)
    for i in range(M):
        plt.subplot(u, v, i + 1)
        plt.hist(X[:, i])
        plt.xlabel(attributeNames[i])
        plt.ylim(0, N / 2)
    plt.show()

def boxplots(X, M, attributeNames):
    plt.figure(figsize=(8, 7))
    plt.boxplot(X)
    plt.xticks(range(1, M+1), attributeNames)
    plt.title('South Africa Heart Disease data set - boxplot')
    plt.show()

def scatterAllAttributes(X, M, attributeNames):
    plt.figure(figsize=(12, 10))
    for m1 in range(M):
        for m2 in range(M):
            plt.subplot(M, M, m1 * M + m2 + 1)
            for c in range(C):
                class_mask = (y == c)
                plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), '.')
                if m1 == M - 1:
                    plt.xlabel(attributeNames[m2])
                else:
                    plt.xticks([])
                if m2 == 0:
                    plt.ylabel(attributeNames[m1])
                else:
                    plt.yticks([])
    plt.legend(classNames)
    plt.show()

def printStatistics(X, M, attributeNames):
    for i in range(0, M):
        print('\nattribute:', attributeNames[i])
        print('Mean:', X[:, i].mean())
        print('Standard Deviation:', X[:, i].std(ddof=1))
        print('Median:',  np.median(X[:, i]))
        print('Range:', X[:, i].max()-X[:, i].min())


# file reader
filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)

raw_data = df.values
X = raw_data[:, 1:-1]

# preprocessing attribute family history
X[X[:, 4] == 'Absent', 4] = 0
X[X[:, 4] == 'Present', 4] = 1

# from object array to float array(else SVD raises error)
X = np.array(X, dtype=np.float64)

y = raw_data[:, -1]
classNames = ['noCHD', 'CHD']

attributeNames = np.asarray(df.columns[1:-1])

N, M = X.shape
C = len(classNames)

plot2attributes(0, 1, X, y, attributeNames, classNames, C)

plotComponentVariance(X)

plot2components(X)

plotHist(X, N, M, attributeNames)

boxplots(X, M, attributeNames)

scatterAllAttributes(X, M, attributeNames)

printStatistics(X, M, attributeNames)