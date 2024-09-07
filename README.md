# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.
2. Type the required program.
3. Print the program.
4. End the program.

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: varsha s
RegisterNumber: 212222220055

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:
![3 1ml](https://user-images.githubusercontent.com/118541549/229344446-20916a74-1d0a-4be2-807c-6c58fb0f9076.png)
![3 2ml](https://user-images.githubusercontent.com/118541549/229344455-49c19898-f99a-4e35-bea5-7da75268f7f5.png)
![3 3 ml](https://user-images.githubusercontent.com/118541549/229344483-5c0674a6-3a0e-482f-85d6-569efd2fdbfc.png)
![3 4ml](https://user-images.githubusercontent.com/118541549/229344487-916c492c-1e5f-455a-a33e-5e4cf327b2e6.png)
![3 5ml](https://user-images.githubusercontent.com/118541549/229344491-01fbf27e-4b73-4ad1-9c2e-227c32982f3e.png)
![3 6ml](https://user-images.githubusercontent.com/118541549/229344499-4f92686d-7493-4ea7-a58f-051927e84916.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
