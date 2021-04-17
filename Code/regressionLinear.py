#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:18:09 2021

@author: Busra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import sklearn.tree
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, legend, ylim, subplot, hist, boxplot, xticks, \
    yticks
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
import os
from toolbox_02450 import rlr_validate
import scipy.stats as st
from scipy.stats import zscore

# Load the SAheart csv data using Panda
filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)
df.describe()
df.dtypes

# Convert 'dataframe' to 'numpy array'
data = df.to_numpy()
coloumn = range(1, 11)

# X - matrix with data without attribute names
X = data[:, coloumn]

# attributenames - array with attribute names
attributeNames = np.asarray(df.columns[coloumn])

# famhistLabel - array with only famhist
famhistLabel = data[:, 5]

# famhistclass - array with the famhist classes from the famhistlabel array
famhistclass = np.unique(famhistLabel)

# famhisttrans - converting absent = 0, present = 1
famhisttrans = dict(zip(famhistclass, range(len(famhistclass))))

# Y = famhist array with 0 and 1 instead of absent/present
famhistarray = np.array([famhisttrans[cl] for cl in famhistLabel])

# remove famhist from df
df = df.drop(df.columns[[5]], axis=1)

# run everything for removing famhist
data = df.to_numpy()
coloumn = range(1, 10)
X = data[:, coloumn]
attributeNames = np.asarray(df.columns[coloumn])
N, M = X.shape

# cNumber = length of famhist class
C = len(famhistclass)

# famhist convert to matrix and transposed
famhistarray = np.asmatrix(famhistarray)
famhistarray = famhistarray.T

# x1 - Updated array with famhist
X1 = np.hstack((X, famhistarray))

# update attribute name
attributeNames = np.append(attributeNames, "famhist")

# X1 converted to array
X1 = np.asarray(X1)

## X3 - NORMALIZATION OF DATA
X2 = X1 - np.ones((N, 1)) * X1.mean(0)
X3 = X2 * (1 / np.std(X2, 0))

# ldlarray - array for data without ldl
ldlarray = np.delete(X1, 2, axis=1)
# ldl - array for normalization of ldl
ldl = X1[:, [2]]
# normalization of ldlarray
X2ldl = ldlarray - np.ones((N, 1)) * ldlarray.mean(0)
X3ldl = X2ldl * (1 / np.std(X2ldl, 0))
attribute = np.delete(attributeNames, 2, axis=0)
Y1ldl = ldl - np.ones((N, 1)) * ldl.mean(0)
Y2ldl = Y1ldl * (1 / np.std(Y1ldl, 0))

# chdarray - array for data without chd
chdarray = np.delete(X1, 8, axis=1)
chd = X1[:, [8]]
X2chd = chdarray - np.ones((N, 1)) * chdarray.mean(0)
X3chd = X2chd * (1 / np.std(X2chd, 0))
Y1chd = chd - np.ones((N, 1)) * chd.mean(0)
Y2chd = Y1chd * (1 / np.std(Y1chd, 0))

# update
X = ldlarray
N, M = X.shape
Y = ldl.squeeze()

# offset attribute
X = zscore(X, ddof=1)
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

attributenames = []
attributenames.append('Offset')
for i in range(len(attribute)):
    attributenames.append(attribute[i])

attributeNames = attributenames
M = M + 1

# PART A

# Regularization parameter lambda

relambda = np.power(10., range(-5, 9))

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, Y2ldl, relambda,
                                                                                                  10)

# Plot
figure(figsize=(12, 8))

subplot(1, 2, 1)
semilogx(relambda, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
xlabel('Lambda')
ylabel('Mean Coefficient')
grid()
legend(attributeNames[1:], loc='best')

subplot(1, 2, 2)
title('Optimal Lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(relambda, train_err_vs_lambda.T, 'b.-', relambda, test_err_vs_lambda.T, 'r.-')
xlabel('Lambda')
ylabel('Squared error (crossvalidation)')
legend(['Train error', 'Validation error'])
grid()

show()

print('Weights as function of lambda:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(mean_w_vs_lambda[m, -1], 2)))

# PART B

X = ldlarray
Y = ldl.squeeze()
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

# Implemtation of two level cross validation
K = 10
CVModel = model_selection.KFold(K, shuffle=True)

# relambda - lambda value
relambda = np.power(10., range(-5, 9))

# Training and test error
ErrorTrain = np.empty((K, 1))
ErrorTest = np.empty((K, 1))
ErrorTrainrlr = np.empty((K, 1))
ErrorTestrlr = np.empty((K, 1))
ErrorTrainNoFeatures = np.empty((K, 1))
ErrorTestNoFeatures = np.empty((K, 1))

wrlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
wnoreg = np.empty((M, K))

# LambdaOpt - Empty array for optimal lambda valuees
LambdaOpt = np.array([])
k = 0

for IndexTrain, IndexTest in CVModel.split(X, Y):
    # extract training and test set for current CV fold
    TrainX = X[IndexTrain]
    TrainY = Y[IndexTrain]
    TestX = X[IndexTest]
    TestY = Y[IndexTest]
    internal_cross_validation = 10

    OptValErr, OptLambda, MeanWvsLambda, ErrorTrainvsLambda, ErrorTestvsLambda = rlr_validate(TrainX, TrainY, relambda,
                                                                                              internal_cross_validation)
    # Save values
    LambdaOpt = np.append(LambdaOpt, OptLambda)

    mu[k, :] = np.mean(TrainX[:, 1:], 0)
    sigma[k, :] = np.std(TrainX[:, 1:], 0)
    TrainX[:, 1:] = (TrainX[:, 1:] - mu[k, :]) / sigma[k, :]
    TestX[:, 1:] = (TestX[:, 1:] - mu[k, :]) / sigma[k, :]
    XtY = TrainX.T @ TrainY
    XtX = TrainX.T @ TrainX

    # Mean squared error - no features
    ErrorTrainNoFeatures[k] = np.square(TrainY - TrainY.mean()).sum(axis=0) / TrainY.shape[0]
    ErrorTestNoFeatures[k] = np.square(TestY - TestY.mean()).sum(axis=0) / TestY.shape[0]

    # Train set - weights
    lambdaI = OptLambda * np.eye(M)
    lambdaI[0, 0] = 0
    wrlr[:, k] = np.linalg.solve(XtX + lambdaI, XtY).squeeze()

    # Mean squared error - regularization
    ErrorTrainrlr[k] = np.square(TrainY - TrainX @ wrlr[:, k]).sum(axis=0) / TrainY.shape[0]
    ErrorTestrlr[k] = np.square(TestY - TestX @ wrlr[:, k]).sum(axis=0) / TestY.shape[0]

    # Train set -  weights
    wnoreg[:, k] = np.linalg.solve(XtX, XtY).squeeze()

    # Mean squared error
    ErrorTrain[k] = np.square(TrainY - TrainX @ wnoreg[:, k]).sum(axis=0) / TrainY.shape[0]
    ErrorTest[k] = np.square(TestY - TestX @ wnoreg[:, k]).sum(axis=0) / TestY.shape[0]

    # Plot cross-validation for last fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 2, 1)
        semilogx(relambda, MeanWvsLambda.T[:, 1:], '.-')  # Don't plot the bias term
        xlabel('Lambda')
        ylabel('Mean Coefficient')
        legend(attributeNames[1:], loc='best')
        grid()

        subplot(1, 2, 2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(relambda, ErrorTrainvsLambda.T, 'b.-', relambda, ErrorTestvsLambda.T, 'r.-')
        xlabel('Lambda')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error', 'Validation error'])
        grid()

    # To inspect the used indices, use these print statements
    print('Cross validation fold {0}/{1}:'.format(k + 1, K))
    print('Train indices: {0}'.format(IndexTrain))
    print('Test indices: {0}\n'.format(IndexTest))

    k += 1
    show()

print('Linear regression without feature selection:')
print('- Training error: {0}'.format(ErrorTrain.mean()))
print('- Test error:     {0}'.format(ErrorTest.mean()))
print('- R^2 train:     {0}'.format((ErrorTestNoFeatures.sum() - ErrorTrain.sum()) / ErrorTrainNoFeatures.sum()))
print('- R^2 test:     {0}\n'.format((ErrorTestNoFeatures.sum() - ErrorTest.sum()) / ErrorTestNoFeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(ErrorTrainrlr.mean()))
print('- Test error:     {0}'.format(ErrorTestrlr.mean()))
print('- R^2 train:     {0}'.format((ErrorTrainNoFeatures.sum() - ErrorTrainrlr.sum()) / ErrorTrainNoFeatures.sum()))
print('- R^2 test:     {0}\n'.format((ErrorTestNoFeatures.sum() - ErrorTestrlr.sum()) / ErrorTestNoFeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(wrlr[m, -1], 2)))

##Confidence interval og p-value
alpha = 0.05
z = ErrorTestrlr - ErrorTestNoFeatures
CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
print(CI, p)












