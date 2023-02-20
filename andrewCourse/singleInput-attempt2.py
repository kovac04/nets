import numpy as np

x = np.array([1,2,3,4])   #inputs
y = np.array([100,200,300,400])    #targets

#implement loss - (1/2m)*sum((wx+b - y)^2)
def forward(w,x,b):
    return w*x+b

def loss(x,y,w,b):  #loss function = (y_pred-y)^2
    m = len(x)
    cost = 0
    for i in range(m):  #do loss for each input
        y_pred = forward(w,x[i],b) 
        cost += (y_pred - y[i])**2  
    return cost/(2*m)  #factor of 2 makes it easier to take
    # the derivative of the cost function when minimizing it 
    # using gradient descent, as the factor of 2 cancels out 
    # when taking the derivative


#implement compute_gradient - (1/m) * sum(x*(wx+b - y))
                            # (1/m) * sum(wx+b - y)
def compute_gradient(x,y,w,b):
    dj_dw = 0       #J = loss function / take derivative, w.r.t. weights
    dj_db = 0       #take derivative w.r.t. bias
    m = len(x)      
    for i in range(m):
        y_pred = forward(w,x[i],b)
        dj_dw += x[i]*(y_pred - y[i]) #dj/dw
        dj_db += (y_pred - y[i])    #dj/db
    return dj_dw/m, dj_db/m
    

#implement gradient_descent   w = w - alpha*dj_dw
                             #b = b - alpha*dj_db
def gradient_descent(x,y,w,b):

    for i in range(10000):
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w = w - 0.01*dj_dw 
        b = b - 0.01*dj_db
    return w,b

#make prediction with trained model
w,b=0,0
new_w,new_b = gradient_descent(x,y,w,b)
print(f"{forward(5,new_w,new_b):3.4}")