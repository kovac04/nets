import numpy as np
import matplotlib as plt

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
w = np.array([1,1])
b = -3

def predict(w,X,b):
    z = np.dot(w,X) + b
    y_pred = 1/(1+np.exp(-z))
    return y_pred

prediction = predict(w,X[0],b)
print(prediction)
# if(prediction >= 0.5):
#     print("cornflakes")

def computeCost(w,X,b,y):   #new cost function
    cost = 0
    for i in range(len(X)):
        z_i = np.dot(w,X[i]) + b    
        y_pred = 1/(1+np.exp(-z_i))
        cost +=  -y[i]*np.log(y_pred) - (1-y[i])*np.log(1-y_pred)
    return cost/len(X)

print(computeCost(w,X,b,y))

