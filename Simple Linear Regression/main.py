import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LinearRegression
# read the data file
dataset = pd.read_csv('Salary_Data.csv')

# select the parameters and the output
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# fill missing data with the mean
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(x)
x = impute.transform(x)

# train split the model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size=.2)

# Linear Regression

regression = LinearRegression()
regression.fit(x_train, y_train)
y_predict = regression.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.show()
