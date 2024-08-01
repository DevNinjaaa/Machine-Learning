import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# read the data
dataset = pd.read_csv('Position_Salaries.csv')

# independent variables
x = dataset.iloc[:, 1:-1].values
# dependant variables
y = dataset.iloc[:, -1].values

# reshape Y
y = y.reshape(len(y), 1)

# feature scaling for x && y
ss_x = StandardScaler()
ss_y = StandardScaler()
x = ss_x.fit_transform(x)
y = ss_y.fit_transform(y).ravel()

# SVR regression with Gaussian radial basis function rbf
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# prediction
print(ss_y.inverse_transform(regressor.predict(ss_x.transform([[6.5]])).reshape(-1, 1)))

# Inverse transform the scaled data to original values
x_inv = ss_x.inverse_transform(x).reshape(-1, 1)
y_inv = ss_y.inverse_transform(y.reshape(-1, 1))

# Predict using the SVR model and inverse transform the predictions
y_pred = regressor.predict(x)
y_pred_inv = ss_y.inverse_transform(y_pred.reshape(-1, 1))

# Plotting the results
plt.scatter(x_inv, y_inv, color='red', label='Actual')
plt.plot(x_inv, y_pred_inv, color='blue', label='SVR Model')
plt.title('Position Level vs Salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()
