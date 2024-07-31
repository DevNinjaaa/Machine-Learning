import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read the data
dataset = pd.read_csv('Position_Salaries.csv')

# independent variables
x = dataset.iloc[:, 1:-1].values
# dependant variables
y = dataset.iloc[:, -1].values

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# polynomial regression
poly_f = PolynomialFeatures(degree=4)
x_poly = poly_f.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)

# #  linear regression vs polynomial regression
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('linear regression')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,poly_reg.predict(x_poly), color='blue')
plt.title('Polynomial regression')
plt.show()

# predicting a new result with linear regression
print(lin_reg.predict([[5.5]]))
# predicting a new result with polynomial regression
print(poly_reg.predict(poly_f.fit_transform([[5.5]])))
