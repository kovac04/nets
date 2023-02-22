import copy, math
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
alpha = 0.5

def forward(w,x,b):
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


def compute_gradient(X,y,w,b):
    m = len(X_train)         #(number of examples)
    n = len(X_train[0])         #(number of features)
    dj_dw = np.zeros((n,))      
    dj_db = 0.
    for i in range(m):   #go through each training example                          
        err = (np.dot(X[i],w)+b - y[i]) #same as (y_pred - y[i]), still doing it for each example
        for j in range(n):    #go through all features                     
            dj_dw[j] += err * X[i, j] #similar to dj_dw += x[i]*(y_pred - y[i]) / gradient formula
                                    #need to add it for all features too, not just examples
        dj_db += err     #dj_db = err/ y_pred-y / formula / just add all of them to get the average
    dj_dw /= m                          
    dj_db /= m                                
        
    return dj_dw, dj_db


def zScoreNormalize(X):
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)

print(compute_gradient(X_train,y_train,w_init,b_init))

def gradient_descent(X,y,w,b):
    J_history = []
    p_history = []
    
    for i in range(100):
        # Calculate the gradient and update the parameters using compute_gradient
        dj_dw, dj_db = compute_gradient(X, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100:      # prevent resource exhaustion 
            J_history.append( cost(X, y, w, b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history #return w and J,w history for graphing
X_norm, mu, sigma = zScoreNormalize(X_train)
w_final, b_final, J_hist, p_hist = gradient_descent(X_norm, y_train, w_init, b_init)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()