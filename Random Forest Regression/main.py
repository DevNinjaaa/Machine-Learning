import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
from sklearn.ensemble import RandomForestRegressor

# read the data
dataset = pd.read_csv('Position_Salaries.csv')

# independent variables
x = dataset.iloc[:, 1:-1].values
# dependant variables
y = dataset.iloc[:, -1].values


regressor= RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Position Level vs Salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()