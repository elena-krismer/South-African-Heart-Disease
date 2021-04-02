# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:08:47 2021

@author: kostas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection


#file reader
filename= '../Data/SAheart.csv'
df = pd.read_csv(filename)


raw_data=df.values
X = raw_data[:,1:-1]

#preprocessing attribute family history
X[X[:,4]=='Absent',4]=0
X[X[:,4]=='Present',4]=1

#from object array to float array
X=np.array(X, dtype=np.float64) 

y = raw_data[:,-1]
classNames = ['noCHD','CHD']

attributeNames = np.asarray(df.columns[1:-1])

N, M = X.shape
C = len(classNames)

K1 = 10

arrayE = []


CVouter = model_selection.KFold(K1,shuffle=True)

for (k1, (par_index, test_index)) in enumerate(CVouter.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k1+1,K1)) 
    
    y_par = y[par_index]
    y_test = y[test_index]
    
    if len(y_par[y_par==0]) > len(y_par[y_par==1]): #find the largest class
        y_est=0
    else:
        y_est=1
        
    misclass_rate = np.sum(y_est != y_test) / float(len(y_test)) #calculate error based on the largest class
    arrayE.append(misclass_rate)

    