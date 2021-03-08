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

#dataset histograms
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

#dataset boxplots
def boxplots(X, M, attributeNames):
    plt.figure(figsize=(8, 7))
    plt.boxplot(X)
    plt.xticks(range(1, M+1), attributeNames)
    plt.title('South Africa Heart Disease data set - boxplot')
    plt.show()

#scatter plot for every pair
def scatterAllAttributes(X, M, attributeNames):
    plt.figure(figsize=(12, 10))
    X = np.delete(X, 4, axis=1)
    attributeNames=np.delete(attributeNames, 4)
    M=M-1
    for m1 in range(M):
        for m2 in range(M):
            plt.subplot(M, M, m1 * M + m2 + 1)
            for c in range(C):
                class_mask = (y == c)
                plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), '.', alpha=0.6)
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

#calculate basic statistics
def printStatistics(X, M, attributeNames):
    for i in range(0, M):
        print('\nattribute:', attributeNames[i])
        print('Mean:', X[:, i].mean())
        print('Standard Deviation:', X[:, i].std(ddof=1))
        print('Median:',  np.median(X[:, i]))
        print('Range:', X[:, i].max()-X[:, i].min())

#Principal Component Analysis
def PCAnalysis(X):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    Y = Y * (1 / np.std(Y, 0))

    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    # variance
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

    # plot attribute coefficients
    i = 0
    j = 1
    plt.figure()
    for att in range(V.shape[1]):
        plt.arrow(0, 0, V[att, i], V[att, j])
        plt.text(V[att, i], V[att, j], attributeNames[att])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)),
             np.sin(np.arange(0, 2 * np.pi, 0.01)));
    plt.title('Attribute coefficients')
    plt.axis('equal')
    plt.show()
    for i in range(4):
        print('PC{}:'.format(i+1))
        print(V[:, i])

    # project to principal component space
    Z = Y @ V

    i = 0
    j = 1
    plt.figure()
    plt.title('Projected Data')
    for c in range(C):
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))
    plt.show()

    i = 0
    j = 1
    k = 2
    plt.figure()
    plt.title('3D Projected Data')
    ax = plt.axes(projection='3d')
    colors = ["blue","orange"]
    for c in range(C):
        class_mask = y == c
        ax.scatter3D(Z[class_mask, i], Z[class_mask, j], Z[class_mask, k], c=colors[c],alpha=0.5)
    plt.legend(classNames)
    ax.set_xlabel('PC{0}'.format(i + 1))
    ax.set_ylabel('PC{0}'.format(j + 1))
    ax.set_zlabel('PC{0}'.format(k + 1))
    plt.show()


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

plotHist(X, N, M, attributeNames)

boxplots(X, M, attributeNames)

scatterAllAttributes(X, M, attributeNames)

printStatistics(X, M, attributeNames)

PCAnalysis(X)

# similarity matrix
df = df.drop('row.names', 1)
corr=df.corr()
print(corr)