import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# read the data
dataset = pd.read_csv('Position_Salaries.csv')

# independent variables
x = dataset.iloc[:, 1:-1].values
# dependant variables
y = dataset.iloc[:, -1].values

# reshape Y

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x, y)
regr_2.fit(x, y)

# prediction
y_1 = regr_1.predict(x)
y_2 = regr_2.predict(x)


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_grid, regr_1.predict(x_grid), color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(x_grid, regr_2.predict(x_grid), color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
