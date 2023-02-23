import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data

#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train, y_train = load_house_data()
b_init = 3.6
w_init = np.array([ 8.9, 3, 3.3, -6.0])
alpha = 1.0e-1

def forward(w,x,b):
    return np.dot(w,x) + b


def cost(x,y,w,b):
    m = len(x)
    cost = 0.0
    for i in range(m):
        y_pred = forward(w,x[i],b) #forward all features together, get pred
        cost+= (y_pred - y[i])**2   #same cost function as before
    return cost/(2*m)

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


def gradient_descent(X,y,w,b):
    J_history = []
    p_history = []
    
    for i in range(1000):
        # Calculate the gradient and update the parameters using compute_gradient
        dj_dw, dj_db = compute_gradient(X, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<1000:      # prevent resource exhaustion 
            J_history.append( cost(X, y, w, b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history #return w,b and J,w history for graphing


X_norm, mu, sigma = zScoreNormalize(X_train) #normalize the inputs
w_norm, b_norm, J_hist, p_hist = gradient_descent(X_norm, y_train, w_init, b_init)
print(f"\nOptimal w and b = {w_norm} , {b_norm}")

x_unknown = [1200, 3, 1, 40]
x_unknown_norm  = (x_unknown - mu) / sigma # z score normalize 
print(f"Predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${forward(w_norm,x_unknown_norm,b_norm)*1000:0.0f}")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()