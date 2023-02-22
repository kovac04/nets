import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Load our data set
x = np.array([0,1,2,3])   #inputs
y = np.array([0,1,2,3])   #target value

#Function to calculate the cost
def loss(x, y, w, b):
   
    m = len(x)
    cost = 0
    
    for i in range(m):
        y_pred = w * x[i] + b
        cost = cost + (y_pred - y[i])**2
    total_cost = (1/(2 * m)) * cost

    return total_cost

def compute_gradient(x, y, w, b): 
    m = len(x)
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        y_pred = w * x[i] + b 
        dj_dw_i = x[i]*(y_pred - y[i]) 
        dj_db_i = y_pred - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha): 
    J_history = []
    p_history = []
    
    for i in range(100):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100:      # prevent resource exhaustion 
            J_history.append( loss(x, y, w, b))
            p_history.append([w,b])
 
    return w, b, J_history, p_history #return w and J,w history for graphing

# initialize parameters
w_init = 0
b_init = 0
tmp_alpha = 0.4
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x ,y, w_init, b_init, tmp_alpha)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")



# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

