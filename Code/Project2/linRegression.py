# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary,rlr_validate


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

lambdas = np.power(10., range(-5, 7))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

# Xst=X - np.ones((N, 1)) * X.mean(axis=0)
# Xst = Xst * (1 / np.std(Xst, 0))

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)


opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas,K)


plt.figure(figsize=(12, 8))  
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()

plt.subplot(1,2,2)                                                                                            
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor (λ)')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Generalization error'])
plt.grid()














