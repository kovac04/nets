import copy, math
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def forward(w,x,b):
    #wx+b
    return np.dot(w,x) + b

x_vec = X_train[0,:]
print("First example prediciton: ",forward(w_init,x_vec,b_init))


def cost(x,y,w,b):
    m = len(x)
    cost = 0.0
    for i in range(m):
        y_pred = forward(w,x[i],b) #forward all features together, get pred
        cost+= (y_pred - y[i])**2   #same cost function as before
    return cost/(2*m)
 
cost1 = cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost1}')


def compute_gradient(x,y,w,b):
    m = len(X_train)         #(number of examples)
    n = len(X_train[0])         #(number of features)
    dj_dw = np.zeros((n,))      
    dj_db = 0.
    for i in range(m):   #go through each training example                          
        err = (forward(w,x[i],b) - y[i]) #same as (y_pred - y[i]), still doing it for each example
        for j in range(n):    #go through all features                     
            dj_dw[j] += err * x[i, j] #similar to dj_dw += x[i]*(y_pred - y[i]) / gradient formula
                                    #need to add it for all features too, not just examples
        dj_db += err     #dj_db = err/ y_pred-y / formula / just add all of them to get the average
    dj_dw = dj_dw / m                          
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


print(compute_gradient(X_train,y_train,w_init,b_init))

#def gradient_descent(X_train,y_train,w_init,b_init):
    
