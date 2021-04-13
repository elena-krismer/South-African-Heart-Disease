from pylab import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns

# file reader
filename = '../Data/SAheart.csv'
df = pd.read_csv(filename)
print(df.describe())

# preprocessing famhist
df["famhist"].replace({"Absent": 0, "Present": 1}, inplace=True)

X = df[["sbp", "tobacco", "famhist", "typea", "adiposity", "obesity", "alcohol", "age", "chd"]]
# use ldl level as predictor variable
y = df["ldl"]

# splitting data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear regression Model
model = LinearRegression()
# train the model
model.fit(X_train, y_train)

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# Apply trained model to make prediction
y_pred = model.predict(X_test)

df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print("Accuracy", model.score(X_test, y_test))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Absolute Error:", np.square(metrics.mean_absolute_error(y_test, y_pred)))

# Scatter plots
sns.scatterplot(y_test, y_pred)
