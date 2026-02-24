import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load dataset
df = sns.load_dataset("tips")

# encode categorical columns
df['sex'] = df['sex'].cat.codes
df['smoker'] = df['smoker'].cat.codes

# create dummy variables
days = pd.get_dummies(df['day'], drop_first=True, dtype=int)
times = pd.get_dummies(df['time'], drop_first=True, dtype=int)

df = pd.concat([df, days, times], axis=1)

# features and target
X = df[['sex','smoker','size','Fri','Sat','Sun','Dinner']]
y = df['tip']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=26
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
predictions = model.predict(X_test)

# evaluation
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("R2:", metrics.r2_score(y_test, predictions))