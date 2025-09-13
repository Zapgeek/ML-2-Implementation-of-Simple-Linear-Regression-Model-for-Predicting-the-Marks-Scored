# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and prepare the data:
      Load the dataset, and split it into independent (X) and dependent (y) variables.
2. Use train_test_split to divide the dataset into training and testing sets.
3. Create a LinearRegression object and fit it with training data (x_train, y_train).
4. Predict using the model on x_test, then calculate errors (MSE, MAE, RMSE) to evaluate performance.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pranav Bhargav M
RegisterNumber: 212224040239
*/
```

```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
print(*Y_pred)
Y_test
print(*Y_test)
```
```
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
<img width="862" height="598" alt="image" src="https://github.com/user-attachments/assets/48abacb9-d8c7-4e9d-b445-577394d3369c" />

<img width="757" height="648" alt="image" src="https://github.com/user-attachments/assets/9aa2137b-2112-4e1b-a5ba-fbee6ead50a4" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
