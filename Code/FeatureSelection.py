import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# file reader
filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)
print(df.describe())

# preprocessing famhist
df["famhist"].replace({"Absent": 0, "Present": 1}, inplace=True)
X = df.drop(columns='ldl')

#X = df[["sbp", "tobacco", "famhist", "typea", "adiposity", "obesity", "alcohol", "age", "chd"]]
#X = df[["adiposity", "obesity", "age"]]
# use ldl level as predictor variable
y = df["ldl"]

cv = KFold(n_splits=10, random_state=0, shuffle=True)
pipeline = make_pipeline(StandardScaler(), LinearRegression())
y_pred = cross_val_predict(pipeline, X, y, cv=cv)
print("\nRMSE: " + str(round(sqrt(mean_squared_error(y, y_pred)), 2)))
print("R_squared: " + str(round(r2_score(y, y_pred), 2)))


# Remove Features with low variance or so
print('\nFilter Features by Variance')
print(X.var())
remove_binary_df = X.drop(columns=['famhist', 'chd'])
y_pred = cross_val_predict(pipeline, remove_binary_df, y,cv=cv)
print("\nRMSE: " + str(round(sqrt(mean_squared_error(y, y_pred)), 2)))
print("R_squared: " + str(round(r2_score(y, y_pred), 2)))


print("Filter Features by Correlation")
fig_dims = (12,8)
fig, ax = plt.subplots(figsize = fig_dims)
sn.heatmap(X.corr(), ax=ax)
#plt.show()

print(abs(df.corr()['ldl']))

# Filtering features according correlation
vals = [0.1, 0.2, 0.3, 0.4]
for val in vals:
    features = abs(df.corr()['ldl'][abs(df.corr()['ldl'])>val].drop('ldl')).index.tolist()
    X=df.drop(columns= 'ldl')
    X=X[features]
    print(features)
    y_pred= cross_val_predict(LinearRegression(), X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y, y_pred)), 2)))
    print("R_squared: " + str(round(r2_score(y, y_pred), 2)))
    print("Mean Absolute Error:", metrics.mean_absolute_error(y, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y, y_pred))
    print("Root Mean Absolute Error:", np.square(metrics.mean_absolute_error(y, y_pred)))
    print("\n")
