# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.

## Program:
```
Program to implement the linear regression using gradient descent.

Developed by: YUGENDARAN.G
RegisterNumber:  212221220063

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('/content/ex1 (2).txt',header=None)

print("Profit Prediction Graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y, theta and generate the cost function in the linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
print("Compute Cost Value:")
computeCost(X,y,theta) #call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta -= descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history
  
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
  
print("Cost function using Gradient Descent:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(/Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit Prediction:")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")  
plt.title("Profit Prediction")
  
def predict(X,theta):
predictions=np.dot(theta.transpose(),X)
return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the Population 35,000:")
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))
  
predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the Population 70,000:")
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
  
```

## Output:
![image](https://user-images.githubusercontent.com/128135616/229772505-9b5e215a-7f9b-4d6a-9c13-83b256c03adb.png)

![image](https://user-images.githubusercontent.com/128135616/229774601-64bf84fa-7c0a-4d97-bf6a-fc72006784fd.png)

![image](https://user-images.githubusercontent.com/128135616/229775286-32cdf640-cddd-4d35-8162-2d864570c1c5.png)

![image](https://user-images.githubusercontent.com/128135616/229775868-28b0321d-38fc-4451-8a09-d6a3868c3fb4.png)

![image](https://user-images.githubusercontent.com/128135616/229776490-eb092d8b-bb7e-4918-8922-fe4c52f2c591.png)

![image](https://user-images.githubusercontent.com/128135616/229782873-cb1b2f5f-179a-4f8b-92e4-0c4525631fab.png)

![image](https://user-images.githubusercontent.com/128135616/229779544-1caedee5-4230-48e0-b303-b33f2f9ce8b1.png)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
