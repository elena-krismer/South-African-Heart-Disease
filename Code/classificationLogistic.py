
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.tree
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, legend, ylim, subplot, hist, boxplot, xticks, \
    yticks
from matplotlib.pylab import figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid
import os
from toolbox_02450 import rlr_validate, rocplot, confmatplot
import scipy.stats as st
from scipy.stats import zscore

filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)

# preprocessing famhist
df["famhist"].replace({"Absent": 0, "Present": 1}, inplace=True)


X = df[["sbp", "ldl", "tobacco", "famhist", "typea", "adiposity", "obesity", "alcohol", "age"]]
#X = df[["adiposity", "obesity", "age"]]
# dependend variable / labelled class
y = df["chd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
K = 20

# removing mean and scaling to unit variance
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Fit regularized logistic regression model to training data to predict
# use different lambda
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    logreg = LogisticRegression(penalty='l2', C=(1 / lambda_interval[k]))
    logreg.fit(X_train, y_train.astype('int'))

    y_train_est = logreg.predict(X_train)
    y_test_est = logreg.predict(X_test)

    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = logreg.coef_[0]
    coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]
print(opt_lambda)

plt.figure(figsize=(8, 8))
# plt.plot(np.log10(lambda_interval), train_error_rate*100)
# plt.plot(np.log10(lambda_interval), test_error_rate*100)
# plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate * 100)
plt.semilogx(lambda_interval, test_error_rate * 100)
plt.semilogx(opt_lambda, min_error * 100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error * 100, 2)) + ' % at 1e' + str(
    np.round(np.log10(opt_lambda), 2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error', 'Test error', 'Test minimum'], loc='upper right')
plt.ylim([0, 4])
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, coefficient_norm, 'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()