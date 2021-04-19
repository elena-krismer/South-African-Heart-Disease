# -*- coding: utf-8 -*-

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.stats.api as sms
import scipy.stats as st
from sklearn.preprocessing import StandardScaler


filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)

# preprocessing famhist
df["famhist"].replace({"Absent": 0, "Present": 1}, inplace=True)


X = df[["sbp", "ldl", "tobacco", "famhist", "typea", "adiposity", "obesity", "alcohol", "age"]]
#X = df[["adiposity", "obesity", "age"]]
# dependend variable / labelled class
y = df["chd"]

# removing mean and scaling to unit variance
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

K1 = 10
K2 = 10

# determine optimal lambda for logistic regression
lambda_interval = np.logspace(-2, 2, 10)
valerrors = np.zeros([len(lambda_interval),K2])
#opt_lambda = 3.72
arrayL = []

arrayE_baseline = []
arrayE_naivebayes = []
arrayE_logreg = []

nb_logreg_pvalue = []
#nb_logreg_pvalue = np.array(nb_logreg_pvalue, dtype=object)
base_logreg_pvalue = []
base_nb_pvalue = []


def correct_classification(y_pred, y_test):
    classifier_arr = np.equal(y_pred, y_test)
    classifier_arr = np.where(classifier_arr == True, 0, 1)
    return classifier_arr


# pairwaise comparison of classifier models
def mc_nemar(classifier_1, classifier_2, model_1, model_2):
    table = confusion_matrix(classifier_1, classifier_2)
    c1, c2 = (model_2 + "- correct"), (model_2 + "- wrong")
    i1, i2 = (model_1 + "- correct"), (model_1 + "- wrong")
    table_pd = pd.DataFrame(table, columns=(c1, c2), index=(i1, i2))
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    alpha = 0.05
    print("Comparison of ", model_1, "and", model_2)
    print(table_pd)
    print("Statistic= %.3f, p-value= %.3f" % (result.statistic, result.pvalue))
    if result.pvalue > alpha:
        print(" -> 0 Hypothesis can not be rejected\n")
    else:
        print(" -> Reject 0 Hypothesis\n")
    return result.pvalue


def split_mcnemar(y_est, y_est_nb, y_est_logreg):
    # Baseline array classifier
    base_mcnemar = correct_classification(y_est, y_test)
    # NB array classfier correct
    nb_mcnemar = correct_classification(y_est_nb, y_test)
    # Logistic Regression array classfier correct
    logreg_mcnemar = correct_classification(y_est_logreg, y_test)

    # Compare Models Pairwaise with McNemar
    print("Pairwaise Compairson of Classifiers with McNemar")
    base_nb_pvalue.append(mc_nemar(base_mcnemar, nb_mcnemar, "Baseline", "Naive Bayes"))
    base_logreg_pvalue.append(mc_nemar(base_mcnemar, logreg_mcnemar, "Baseline", "Logistic Regression"))
    nb_logreg_pvalue.append(mc_nemar(nb_mcnemar, logreg_mcnemar, "Naive Bayes", "Logistic Regression"))


CVouter = model_selection.KFold(K1, shuffle=True)
for (k1, (par_index, test_index)) in enumerate(CVouter.split(X, y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k1 + 1, K1))

    # Extract training and test set for current CV fold, convert to tensors
    X_par = X[par_index, :].astype('float64')
    y_par = y[par_index].astype('int')
    X_test = X[test_index, :]
    y_test = y[test_index]

    # removing mean and scaling to unit variance
    # sc = StandardScaler()
    # sc.fit(X_par)
    # X_par = sc.transform(X_par)
    # X_test = sc.transform(X_test)

    # baseline
    if len(y_par[y_par == 0]) > len(y_par[y_par == 1]):  # find the largest class
        y_est = 0
    else:
        y_est = 1

    misclass_rate = np.sum(y_est != y_test) / float(len(y_test))  # calculate error based on the largest class
    arrayE_baseline.append(misclass_rate)
    
    CVinner=model_selection.KFold(K2,shuffle=True)   
    for (k2, (train_index, val_index)) in enumerate(CVinner.split(X_par,y_par)): 
        
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_val = X[val_index,:]
        y_val = y[val_index]
        
        test_error_rate = np.zeros(len(lambda_interval))
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(C=1/lambda_interval[k], max_iter=100000 )
            
            mdl.fit(X_train, y_train)
        
            y_test_est = mdl.predict(X_val).T
            
            valerrors[k,k2] = np.sum(y_test_est != y_val) / len(y_val)
            
            
        
    EgenS=np.sum(valerrors,1)*X_val.shape[0]/X_par.shape[0]  #generalization error for each model
    opt_lambda_idx=np.argmin(EgenS)
    opt_lambda = lambda_interval[opt_lambda_idx]
    arrayL.append(opt_lambda)
    
    
    # build naive bayes classifier
    nb_clf = GaussianNB()
    # train model
    nb_clf.fit(X_par, y_par)
    # Apply trained model to make prediction
    y_est_nb = nb_clf.predict(X_test)
    misclass_rate_nb = np.sum(y_est_nb != y_test) / float(len(y_test))
    arrayE_naivebayes.append(misclass_rate_nb)

    # logistic regression
    logreg_clf = LogisticRegression(C=(1/opt_lambda), max_iter=100000)
    logreg_clf.fit(X_par, y_par)
    y_est_logreg = logreg_clf.predict(X_test)
    misclass_rate_logreg = np.sum(y_est_logreg != y_test) / float(len(y_test))
    arrayE_logreg.append(misclass_rate_logreg)

    # compare model output with mc-nemar
    split_mcnemar(y_est, y_est_nb, y_est_logreg)

print('Error rate Baseline:', np.mean(arrayE_baseline))
print('Error rate Naive Bayes: ',np.mean(arrayE_naivebayes))
print('Error rate Logistic Regression: ', np.mean(arrayE_logreg))

K = np.arange(1, 11)
error_df =pd.DataFrame({"K": K,"Naive Bayes": arrayE_naivebayes,
                        "Logistic Regression": arrayE_logreg, "Baseline": arrayE_baseline})
print("\n")
print('NB vs. LogReg pvalue: ', np.mean(nb_logreg_pvalue),
      'CI:',st.t.interval(0.95, len(nb_logreg_pvalue)-1,
                    loc=np.mean(nb_logreg_pvalue), scale=st.sem(nb_logreg_pvalue)))
print('NB vs. Base pvalue: ', np.mean(base_nb_pvalue),
      'CI:',st.t.interval(0.95, len(base_nb_pvalue)-1,
                    loc=np.mean(base_nb_pvalue), scale=st.sem(base_nb_pvalue)))
print('LogReg vs. Base pvalue: ', np.mean(base_logreg_pvalue),
      'CI:', st.t.interval(0.95, len(base_logreg_pvalue), scale=st.sem(base_logreg_pvalue)))
print("\n")
print(error_df)

count = (y == 0).sum()
count1 = (y == 1).sum()
print(count, count1)

print('opt λ= ', arrayL)