# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries, load the dataset, remove unwanted columns, and encode categorical features.
2.Separate features and target, standardize the data, and split into training and testing sets.
3.Create the SGD Regressor model, train it using the training data, and make predictions.
4.Evaluate using MSE, MAE, R² and plot actual vs predicted values. 

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: ARTHI S
RegisterNumber:  212225220011
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment (1) (5).csv')
print(data.head())
print(data.info())
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
print('Name: ARTHI S')
print('Reg No:212225220011')
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
```
## Output:

<img width="682" height="889" alt="Screenshot 2026-03-28 183857" src="https://github.com/user-attachments/assets/005b899c-19dc-4082-9e28-091242c44675" />
<img width="668" height="601" alt="Screenshot 2026-03-28 183909" src="https://github.com/user-attachments/assets/5cdf950d-c649-4879-a666-b9ae5b9692fd" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
