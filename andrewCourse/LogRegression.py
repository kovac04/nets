import numpy as np
import matplotlib.pyplot as plt
import copy

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
w = np.array([2,2])
b = 4
alpha = 0.1
lamb = 0.00001

def forward(w,X,b):
    z = np.dot(w,X) + b
    y_pred = 1/(1+np.exp(-z))
    return y_pred


def computeCost(w,X,b,y):   #new cost function
    cost = 0
    m = len(X)

    for i in range(len(X)):    
        y_pred = forward(w,X[i],b)
        cost +=  -(y[i]*np.log(y_pred) + (1-y[i])*np.log(1-y_pred)) #normal cost function

    cost/=m

    reg_cost = 0                #(λ/2m)*∑(Wj)^2 regularize all weights
    for j in range(len(w)):
        reg_cost += (w[j]**2)
    reg_cost *= (lamb/(2*m)) 

    total_cost = cost + reg_cost
    return total_cost

def computeGradient(w,X,b,y):
    dj_dw = np.zeros((len(X[0]),))
    dj_db = 0.0
    for i in range(len(X)):
        y_pred = forward(w,X[i],b)
        for j in range(len(X[i])):
            dj_dw[j] += X[i,j]*(y_pred - y[i])
        dj_db += (y_pred - y[i])

    dj_dw/=len(X)
    dj_db/=len(X)
    for j in range(len(w)):         #regularized
        dj_dw[j] += w[j]*(lamb/len(X))      

    return dj_dw,dj_db

def gradientDescent(w,X,b,y):
    w_deep= copy.deepcopy(w)
    b_deep = b
    for i in range(10000):
         dj_dw,dj_db = computeGradient(w_deep,X,b_deep,y)
         w_deep = w_deep - alpha*dj_dw
         b_deep = b_deep - alpha*dj_db
    return w_deep,b_deep

w_new, b_new = gradientDescent(w,X,b,y) 

print(f"Optimal w = {w_new}\nOptimal b = {b_new}")
old_cost = computeCost(w,X,b,y)
new_cost = computeCost(w_new,X,b_new,y)
print(f"\nOld cost = {old_cost}\nNew cost = {new_cost}")

predict = forward(w_new,[3.2,0.9],b_new)
print(f"\n{predict}")


