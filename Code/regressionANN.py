# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:53:51 2021

@author: kostas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch

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
Xcols=list(range(0,yIndex)) + list(range(yIndex+1,len(attributeNames)))
X=X[:,Xcols]



N, M = X.shape
C = 2

K1 = 10
K2 = 10

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

arrayE = []
arrayH = []
valerrors = np.zeros([S,K2])

CVouter = model_selection.KFold(K1,shuffle=True)
for (k1, (par_index, test_index)) in enumerate(CVouter.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k1+1,K1))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_par = torch.Tensor(X[par_index,:])
    y_par = torch.Tensor(y[par_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    
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
                                                               X=X_par,
                                                               y=y_par,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)        
    # Predict values on test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    Etest = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    print("Best h ={}".format(h[lowestIndex]))
    print("Current E={}".format(Etest))
    arrayH.append(h[lowestIndex])
    arrayE.append(Etest[0])
    
print("\nh = {}".format(arrayH))
print("Etest = {}".format(arrayE))