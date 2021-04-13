import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

########################
# Supervised Learning  #
# Classification       #
# Naive Bayes          #
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

# splitting data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build naive bayes classifier
nb_clf = GaussianNB()
# train model
nb_clf.fit(X_train, y_train)
# Apply trained model to make prediction
y_pred = nb_clf.predict(X_test)

# Confusion Matrix, Sensitivity and Specifity
true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()
specificity = true_neg / (true_neg + false_pos)
sensitivity = true_pos / (true_pos + false_neg)

print("Classification - Naive Bayes \n\nAccuracy:", nb_clf.score(X_test, y_test),
      "\nSpecificity:", specificity, "\nSensitivity:", sensitivity,
      "\n\nMean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred),
      "\nMean Squared Error:", metrics.mean_squared_error(y_test, y_pred),
      "\nRoot Mean Absolute Error:",
      np.square(metrics.mean_absolute_error(y_test, y_pred)))
