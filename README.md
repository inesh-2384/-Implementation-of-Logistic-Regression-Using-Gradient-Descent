# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: START

STEP 2: Import the data file and import numpy, matplotlib and scipy.

STEP 3: Visulaize the data and define the sigmoid function, cost function and gradient descent.

STEP 4: Plot the decision boundary .

STEP 5: Calculate the y-prediction.

STEP 6: STOP
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: INESH N 
RegisterNumber: 212223220036

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```
## Output:

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/1b517e96-f207-4f2b-8edc-001dbfc8857c)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/4c0cc6a1-a37b-4074-9e3f-f340fddd8bee)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/dc5a7fcb-ba8e-4add-9f04-e65805f22310)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/7fa94896-6709-4b35-b582-d36fe86675fe)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/aa1fa0db-0cb4-4abb-9b17-fb0cbf8cd850)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/a50a483b-5ff7-4bb6-8c15-7c6086f9f382)

![image](https://github.com/Jaiganesh235/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118657189/3a8e10cb-9a59-40a6-a52e-8005044a0223)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
