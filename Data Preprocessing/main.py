import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read the data file
dataset = pd.read_csv('Data.csv')

# select the parameters and the output
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# fill the missing values
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(x[:, 1:3])
x[:, 1:3] = impute.transform(x[:, 1:3])

# encode the country column
columnT = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnT.fit_transform(x))

# encode the result("yes","no") -> 1,0
le = LabelEncoder()
y = le.fit_transform(y)

# train split the model
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size=.2)

# Feature scaling
sc = StandardScaler()
x_train[:, 3:]= sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])

print(x_train)
print(x_test)
