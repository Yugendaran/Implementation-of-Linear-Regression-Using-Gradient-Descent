# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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


```

## Output:
![image](https://user-images.githubusercontent.com/128135616/229772505-9b5e215a-7f9b-4d6a-9c13-83b256c03adb.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
