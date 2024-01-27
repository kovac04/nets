import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1,2,3,4])   #inputs
y = np.array([100,200,300,400])    #targets

def cost(x,y,w,b):
    # implement function cost - (1/2m)*(w*x[i]+b - y[i])^2
    cost = 0
    for i in range(len(x)): #access each x input
        y_pred = w*x[i]+b   #make the prediction for that x input
        cost = cost +(y_pred - y[i]) ** 2   #calculate the cost for that x input and add it to the previous one
    total_cost = (1/(2*len(x)))*cost    #divide it by number of inputs *2(makes it neater later) = return average cost for this w and b
    return total_cost

def compute_gradient(x,y,w,b):
    #implement equation dj_db - (1/m)*sum(x[i]*(w*x[i]+b))
    #implement equation dj_dw - (1/m)*sum(w*x[i]+b)
    dj_dw,dj_db = 0,0 #derivatives of loss function with respect to w and b
    for i in range(len(x)):
        y_pred = w*x[i]+b   #make the prediction for accessed x input
        dj_dw_i = x[i] * (y_pred - y[i])    #calculate derivative for that x input with respect to w
        dj_db_i = (y_pred - y[i])   #calculate derivative for that x input with respect to b
        dj_db +=dj_db_i #add it to previous derivative of previous x input
        dj_dw +=dj_dw_i #add it to previous derivative of previous x input
    dj_dw / len(x) #get average dj_dw
    dj_db / len(x) #get average dj_db
    return dj_dw,dj_db

def gradient_descent(x,y,w,b):
    J_history = []
    p_history = []
    for i in range(20000): # num of iterations 
        dj_dw,dj_db = compute_gradient(x,y,w,b) #take average gradients for some w and b
        w_temp = w - alpha*dj_dw    
        b_temp = b - alpha*dj_db 
        w = w_temp  #update w
        b = b_temp  #update b
        J_history.append(cost(x,y,w,b))
        p_history.append([w,b])
    return w,b,J_history,p_history

w,b,alpha=0,0,0.01
w,b,J_history,p_history = gradient_descent(x,y,w,b)
print(f"w = {w:8.4f}\nb = {b:8.4f}")
print(f"{w*5+b:8.4f}")


#plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_history[:100])
ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()


