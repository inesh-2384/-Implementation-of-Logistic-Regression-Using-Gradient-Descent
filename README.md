# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights, learning rate, and iterations.
2. Compute predictions using the sigmoid function on weighted inputs.
3. Update weights by applying gradient descent to minimize the cost function.
4. Repeat until convergence, then use final weights to classify and evaluate accuracy.

## Program:

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Inesh N
RegisterNumber:  2122232220036 
*/

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Encode target
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Select numerical features
X = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].values
y = data['status'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    n = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = -(1/n) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
    return cost

# Gradient Descent
def gradient_descent(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []

    for i in range(epochs):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / n_samples
        weights -= lr * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
        
    return weights, cost_history

# Add bias term
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Train model
weights, cost_history = gradient_descent(X_train_bias, y_train, lr=0.1, epochs=10000)

print("Final Weights:", weights)
print("Final Cost:", cost_history[-1])

# Prediction
y_pred = sigmoid(np.dot(X_test_bias, weights))
y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred]

# Accuracy
accuracy = np.mean(y_pred_class == y_test)
print("Accuracy:", accuracy)

# ---------------------------------------------------------
# Plot 1: Cost vs Iterations
plt.figure(figsize=(6,4))
plt.plot(range(len(cost_history)), cost_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

# ---------------------------------------------------------
# Plot 2: Decision Boundary (using 2 features only for visualization)
X_vis = data[['ssc_p', 'hsc_p']].values
y_vis = data['status'].values

# Normalize for plotting
scaler_vis = StandardScaler()
X_vis = scaler_vis.fit_transform(X_vis)

# Add bias
X_vis_bias = np.c_[np.ones((X_vis.shape[0], 1)), X_vis]

# Train only on 2 features for visualization
weights_vis, _ = gradient_descent(X_vis_bias, y_vis, lr=0.1, epochs=5000)

# Plot points
plt.figure(figsize=(6,4))
plt.scatter(X_vis[y_vis==0,0], X_vis[y_vis==0,1], color='red', label='Not Placed')
plt.scatter(X_vis[y_vis==1,0], X_vis[y_vis==1,1], color='blue', label='Placed')

# Decision boundary line
x_values = np.array([min(X_vis[:,0])-1, max(X_vis[:,0])+1])
y_values = -(weights_vis[0] + weights_vis[1]*x_values) / weights_vis[2]
plt.plot(x_values, y_values, label="Decision Boundary")

plt.xlabel("SSC % (scaled)")
plt.ylabel("HSC % (scaled)")
plt.legend()
plt.title("Decision Boundary for Logistic Regression")
plt.show()
```

## Output:
<img width="865" height="78" alt="image" src="https://github.com/user-attachments/assets/8c98bbef-ce5c-4a93-bbf4-0260b7a06857" />

<img width="688" height="502" alt="image" src="https://github.com/user-attachments/assets/9bb062f8-5f4a-4511-92d4-bfb557a1fa39" />

<img width="679" height="495" alt="image" src="https://github.com/user-attachments/assets/f6a1cd61-8e6d-45df-afff-3fa2362dec50" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

