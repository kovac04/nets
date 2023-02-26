import numpy as np
import matplotlib as plt

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
w = np.array([2.,3.])
b = 1

def forward(w,X,b):
    z = np.dot(w,X) + b
    y_pred = 1/(1+np.exp(-z))
    return y_pred


def computeCost(w,X,b,y):   #new cost function
    cost = 0
    for i in range(len(X)):    
        y_pred = forward(w,X[i],b)
        cost +=  -y[i]*np.log(y_pred) - (1-y[i])*np.log(1-y_pred)
    return cost/len(X)

print(computeCost(w,X,b,y))

def computeGradient(w,X,b,y):
    dj_dw = np.zeros((len(X[0]),))
    dj_db = 0
    for i in range(len(X)):
        y_pred = forward(w,X[i],b)
        err = (y_pred - y[i])
        for j in range(len(X[i])):
            dj_dw[j] += X[i,j]*err
        dj_db += err

    return dj_dw/len(X),dj_db/len(X)

dj_dw, dj_db = computeGradient(w,X,b,y)
print(f"dj_dw: {dj_dw.tolist()}" )
print(f"dj_db: {dj_db}" )


