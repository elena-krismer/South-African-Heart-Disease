# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:52:43 2021

@author: kosta
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary,rlr_validate
import torch
import sklearn.linear_model as lm

filename= '../Data/SAheart.csv'
df = pd.read_csv(filename)

raw_data=df.values
X = raw_data[:,1:-1]

#preprocessing attribute family history
X[X[:,4]=='Absent',4]=0
X[X[:,4]=='Present',4]=1

#from object array to float array(else SVD raises error)
X=np.array(X, dtype=np.float64) 
attributeNames = np.asarray(df.columns[1:-1])

yIndex=2  # pick regression target attribute
y = X[:,[yIndex]] 
ylin = X[:,yIndex] 
Xcols=list(range(0,yIndex)) + list(range(yIndex+1,len(attributeNames)))
X=X[:,Xcols]

N, M = X.shape
C = 2

K1 = 5
K2 = 5


lambdas = np.power(10., range(-5, 7))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

Error_train = np.empty((K1,1))
Error_test = np.empty((K1,1))
Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))




maxh = 4

h = range(1,maxh+1) # different hidden units values
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

S=len(h)  # number of models
models=[] # list for the models with different hidden units
for i in range(0,S):  
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, h[i]), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(h[i], 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
    models.append(model)

loss_fn = torch.nn.MSELoss() #  mean-squared-error loss

arrayEANN = []
arrayElr = []
arrayEbase = []
arrayH = []
arrayL = []
valerrors = np.zeros([S,K2])

CVouter = model_selection.KFold(K1,shuffle=True)
for (k1, (par_index, test_index)) in enumerate(CVouter.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k1+1,K1))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_par = X[par_index,:]
    y_par = ylin[par_index]
    X_test = X[test_index,:]
    y_test = ylin[test_index]
    X_parTensor = torch.Tensor(X[par_index,:])
    y_parTensor = torch.Tensor(y[par_index])
    X_testTensor = torch.Tensor(X[test_index,:])
    y_testTensor = torch.Tensor(y[test_index])
    
    
    
    ################# baseline ####################
    
    m = lm.LinearRegression(fit_intercept=True).fit(X_par, y_par)
    Error_test[k1] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]
    
    arrayEbase.append(Error_test[k1][0])
    
    ############# linear regression ###############
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_par, y_par, lambdas,K2)

            
    mu[k1, :] = np.mean(X_par[:, 1:], 0)
    sigma[k1, :] = np.std(X_par[:, 1:], 0)
    
    X_par[:, 1:] = (X_par[:, 1:] - mu[k1, :] ) / sigma[k1, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k1, :] ) / sigma[k1, :] 
    
    Xty = X_par.T @ y_par
    XtX = X_par.T @ X_par
    

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k1] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k1] = np.square(y_par-X_par @ w_rlr[:,k1]).sum(axis=0)/y_par.shape[0]
    Error_test_rlr[k1] = np.square(y_test-X_test @ w_rlr[:,k1]).sum(axis=0)/y_test.shape[0]
    
    arrayElr.append(Error_test_rlr[k1][0])
    arrayL.append(opt_lambda)
    
    ######### neural network ##########
    CVinner=model_selection.KFold(K2,shuffle=True)   
    for (k2, (train_index, val_index)) in enumerate(CVinner.split(X_par,y_par)): 
        
        
        
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_val = torch.Tensor(X[val_index,:])
        y_val = torch.Tensor(y[val_index])
        for s in range(0,S):
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(models[s],
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
    
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Predict values on val set
            y_val_est = net(X_val)
            
            # Determine errors 
            se = (y_val_est.float() - y_val.float())**2 # squared error
            mse = (sum(se).type(torch.float) / len(y_val)).data.numpy() #mean
            #errors.append(mse) # store error rate for current CV fold
            
            valerrors[s,k2]=mse
            
    EgenS=np.sum(valerrors,1)*X_val.shape[0]/X_par.shape[0]  #generalization error for each model
    lowestIndex=np.argmin(EgenS)  # index of the model with the lowest error
            
    net, final_loss, learning_curve = train_neural_net(models[lowestIndex],
                                                               loss_fn,
                                                               X=X_parTensor,
                                                               y=y_parTensor,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)        
    # Predict values on test set
    y_test_est = net(X_testTensor)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_testTensor.float())**2 # squared error
    Etest = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    print("Best h ={}".format(h[lowestIndex]))
    print("Current E={}".format(Etest))
    arrayH.append(h[lowestIndex])
    arrayEANN.append(Etest[0])
    
print("\nh = {}".format(arrayH))
print("EtestANN = {}".format(arrayEANN))
print("\nλ = {}".format(arrayL))
print("Etest linear regression = {}".format(arrayElr))
print("Etest baseline = {}".format(arrayEbase))        