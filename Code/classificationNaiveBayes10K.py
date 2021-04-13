from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import pandas as pd
import numpy as np

########################
# Supervised Learning  #
# Classification       #
# Naive Bayes          #
# 10K Cross Validation #
########################

# Elena Krismer

# NB uses the theory of probability to carry out the classification of data
# As a result, the most important assumption here is the independence of the predicting variable

# file reader
filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)

# preprocessing famhist
df["famhist"].replace({"Absent": 0, "Present": 1}, inplace=True)

# assume that all features are mutually independent
X = df[["sbp", "ldl", "tobacco", "famhist", "typea", "adiposity", "obesity", "alcohol", "age"]]

# dependend variable / labelled class
y = df["chd"]

# 10 K Crossvalidation
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# build naive bayes classifier
nb_clf = GaussianNB()

# estimate accuracy
accuracy_array = cross_val_score(nb_clf, X, y,cv=k_fold, n_jobs=1)
accuracy_avg = sum(accuracy_array)/10

print("Classification - Naive Bayes")
print("Accuracy - 10K Cross Validation")
print("Mean Accuracy:", accuracy_avg)
print("All results:", accuracy_array)


