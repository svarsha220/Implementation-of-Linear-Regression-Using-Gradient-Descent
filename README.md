## EX3:Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import necessary libraries.
2. Define the linear regression function that:
      Adds a bias term to the feature matrix.
   
      Initializes weights (theta) to zero.
   
      Iteratively updates theta using gradient descent based on predictions and errors.
   
4. Load the dataset using pandas.
5. Extract features (X) and target (y) from the dataset and convert them to floats.
6. Scale the features and target using StandardScaler.
7. Train the linear regression model using the defined function to compute theta.
8. Scale new data and use the computed theta to predict the target value.
9. Inverse transform the prediction to obtain the actual value.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VARSHA S
RegisterNumber: 212222220055
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)
        
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:

### DATASET:

![image](https://github.com/user-attachments/assets/44fa6415-1b09-416f-8881-0f18d3cf42f8)

## VALUE OF X:

![image](https://github.com/user-attachments/assets/10d620e2-2866-46c6-9da4-8499eaab72c4)

## VALUE OF X1_SCALED:

![image](https://github.com/user-attachments/assets/121f1f19-a0bb-468d-9028-7544d3b19978)

## PREDICTED VALUE:

![image](https://github.com/user-attachments/assets/4bb22d08-d9bf-4c69-a635-a0dc5990a758)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
