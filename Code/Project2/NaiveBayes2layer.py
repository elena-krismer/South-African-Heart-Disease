# -*- coding: utf-8 -*-
# Elena Krismer
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Naive Bayes Cross validation
# 2 fold
# determining alpha

filename = '../../Data/SAheart.csv'
df = pd.read_csv(filename)

raw_data = df.values
X = raw_data[:, 1:-1]

# preprocessing attribute family history
X[X[:, 4] == 'Absent', 4] = 0
X[X[:, 4] == 'Present', 4] = 1

# from object array to float array
X = np.array(X, dtype=np.float64)
X = OneHotEncoder().fit_transform(X=X)
y = raw_data[:, -1]
y = y.squeeze()
classNames = ['noCHD', 'CHD']

attributeNames = np.asarray(df.columns[1:-1])

N, M = X.shape
C = len(classNames)
K1 = 10
K2 = 10

# naive bayes parameteer
alpha= [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5 ]
#alpha = np.logspace(0.01, 10, 10)


array_alpha = []
arrayE_naivebayes = []

value_arr = []
position = []


CVouter = model_selection.KFold(K1, shuffle=True)
for (k1, (par_index, test_index)) in enumerate(CVouter.split(X, y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k1 + 1, K1))
    # Extract training and test set for current CV fold, convert to tensors
    X_par = X[par_index, :].astype('float64')
    y_par = y[par_index].astype('int')
    X_test = X[test_index, :]
    y_test = y[test_index]

    CVinner = model_selection.KFold(K2, shuffle=True)
    for (k2, (train_index, val_index)) in enumerate(CVinner.split(X_par, y_par)):

        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[val_index, :]
        y_val = y[val_index]
        a = []
        for al in alpha:
            nb_clf = MultinomialNB(alpha=al, fit_prior=True)
            # train model
            nb_clf.fit(X_par.toarray(), y_par)
            # Apply trained model to make prediction
            y_est_nb = nb_clf.predict(X_test)
            misclass_rate_nb = np.sum(y_est_nb != y_test) / float(len(y_test))
            arrayE_naivebayes.append(misclass_rate_nb)
            a.append(misclass_rate_nb)

            # print(k, '  ', misclass_rate_nb)
    value_arr.append(np.min(a))
    position.append(np.argmin(a))

print("Optimal Alpha")
print(value_arr)
for i in position:
    print(alpha[i])
