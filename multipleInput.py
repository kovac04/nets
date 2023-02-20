import copy, math
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
#plt.style.use('./deeplearning.mplstyle')
#np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def forward(w,x,b):
    #wx+b
    return np.dot(w,x) + b

x_vec = X_train[0,:]
#print(forward(w_init,x_vec,b_init))


def cost(x,y,w,b):
    m = len(x)
    cost = 0.0
    for i in range(m):
        y_pred = forward(w,x[i],b)
        cost+= (y_pred - y[i])**2
    cost/=2*m
    return cost
 
cost1 = cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost1}')


def compute_gradient(x,y,w,b):
    m = len(X_train)         #(number of examples)
    n = len(X_train[0])         #(number of features)
    dj_dw = np.zeros((n,))      
    dj_db = 0.

    for i in range(m):   #go through each example                          
        err = (np.dot(x[i], w) + b) - y[i] 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * x[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


print(compute_gradient(X_train,y_train,w_init,b_init))

#def gradient_descent(X_train,y_train,w_init,b_init):
    
