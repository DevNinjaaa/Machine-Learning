import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read the data
dataset = pd.read_csv('50_Startups.csv')

# independent variables
x = dataset.iloc[:, :-1].values

# dependant variables
y = dataset.iloc[:, -1].values

# dummy variables encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# simple test for the model check predictions
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
