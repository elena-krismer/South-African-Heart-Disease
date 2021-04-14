import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# ADD BASELINE AND LOGISTIC REGRESSION

# Elena Krismer

# compares predictet values with result
# correct prediction = True = 1
# returns array True = 1, False = 0
def correct_classification(y_pred, y_test):
    classifier_arr = np.equal(y_pred, y_test)
    classifier_arr = np.where(classifier_arr == True, 1, 0)
    return classifier_arr


# pairwaise comparison of classifier models
def mc_nemar(classifier_1, classifier_2, model_1, model_2):
    table = confusion_matrix(classifier_1, classifier_2)
    c1, c2 = (model_2 + "- correct"), (model_2 + "- wrong")
    i1, i2 = (model_1 + "- correct"), (model_1 + "- wrong")
    table_pd = pd.DataFrame(table, columns=(c1, c2), index=(i1, i2))
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    alpha = 0.5
    print("Comparison of ", model_1, "and", model_2)
    print(table_pd)
    print("Statistic= %.3f, p-value= %.3f" % (result.statistic, result.pvalue))
    if result.pvalue > alpha:
        print(" -> 0 Hypothesis can not be rejected\n")
    else:
        print(" -> Reject 0 Hypothesis\n")


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

# Baseline
# REPLACE
nb_clf = GaussianNB()
# train model
nb_clf.fit(X_train, y_train)
# Apply trained model to make prediction
base_y_pred = nb_clf.predict(X_test)

# build naive bayes classifier
nb_clf = GaussianNB()
# train model
nb_clf.fit(X_train, y_train)
# Apply trained model to make prediction
nb_y_pred = nb_clf.predict(X_test)

# logistic regression
# REPLACE
nb_clf = GaussianNB()
# train model
nb_clf.fit(X_train, y_train)
# Apply trained model to make prediction
logreg_y_pred = nb_clf.predict(X_test)

# Baseline array classifier
base_mcnemar = correct_classification(base_y_pred, y_test)
# NB array classfier correct
nb_mcnemar = correct_classification(nb_y_pred, y_test)
# Logistic Regression array classfier correct
logreg_mcnemar = correct_classification(logreg_y_pred, y_test)

# Compare Models Pairwaise with McNemar
print("Pairwaise Compairson of Classifiers with McNemar")
mc_nemar(base_mcnemar, nb_mcnemar, "Baseline", "Naive Bayes")
mc_nemar(base_mcnemar, logreg_mcnemar, "Baseline", "Logistic Regression")
mc_nemar(nb_mcnemar, logreg_mcnemar, "Naive Bayes", "Logistic Regression")
