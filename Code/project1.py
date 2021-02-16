import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
dat = pd.read_csv('http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data', sep=',')
#Scatter plot 2 attributes against each other
def plot2attributes(att1,att2,X,y,attributeNames,classNames,C):
    plt.figure()
    plt.title('SA heart disease data')
    for c in range(C):
        class_mask = y==c
        plt.plot(X[class_mask,att1], X[class_mask,att2], 'o',alpha=.3)
    plt.legend(classNames)
    plt.xlabel(attributeNames[att1])
    plt.ylabel(attributeNames[att2])
    plt.show()

#Plot component variance
def plotComponentVariance(X):
    Y = X - np.ones((N,1))*X.mean(axis=0)
    #Y=Y*(1/np.std(Y,0))
    
    U,S,V = svd(Y,full_matrices=False)
    rho = (S*S) / (S*S).sum()
    
    threshold = 0.9
    
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()

#Scatter plot 2 components
def plot2components(X):   
    Y = X - np.ones((N,1))*X.mean(axis=0)
    #Y=Y*(1/np.std(Y,0))
    
    U,S,V = svd(Y,full_matrices=False)
     
    V = V.T
    Z = Y @ V
    
    i = 0
    j = 1
    
    # Plot PCA of the data
    plt.figure()
    plt.title('NanoNose data: PCA')
    #Z = array(Z)
    for c in range(C):
        class_mask = y==c
        plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))
    plt.show()


#file reader
filename= '../Data/SAheart.csv'
df = pd.read_csv(filename)

raw_data=df.values
X = raw_data[:,1:-1]

#preprocessing attribute family history
X[X[:,4]=='Absent',4]=0
X[X[:,4]=='Present',4]=1

#from object array to float array(else SVD raises error)
X=np.array(X, dtype=np.float64) 

y = raw_data[:,-1]
classNames = ['noCHD','CHD']

attributeNames = np.asarray(df.columns[1:-1])

N, M = X.shape
C = len(classNames)



plot2attributes(0,1,X,y,attributeNames,classNames,C)

plotComponentVariance(X)

plot2components(X)





