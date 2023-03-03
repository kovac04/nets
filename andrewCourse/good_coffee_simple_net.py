import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as tc

X_np = np.array([[185.32,12.69],
 [259.92,11.87],
 [231.01,14.41],
 [175.37,11.72],
 [187.12,14.13],                                        #Data
 [225.91,12.1 ],
 [208.41,14.18],
 [207.08,14.03],
 [280.6,14.23],
 [202.81,12.25],
 [196.7,13.54],
 [270.31,14.6 ],
 [192.95,15.2 ],
 [213.57,14.28],
 [164.47,11.92],
 [177.26,15.04],
 [241.77,14.9 ],
 [237.,13.13],
 [219.74,13.87],
 [266.39,13.25]])
Y_np = np.array([[1.,0.,0.,0.,1.,1.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])

X_tensor = tc.from_numpy(X_np).float()      #Convert input to tensor so you can normalize
layer_norm = nn.LayerNorm([20,2], eps=1e-05, elementwise_affine=True)
norm_X_tensor = layer_norm(X_tensor)
norm_X_np = norm_X_tensor.detach().numpy()           #convert normalized inputs back to numpy

def myLinear(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)                         
    for j in range(units):                      #units = columns of 'W' matrix      
        w = W[:,j]                                       #go through columns of 'W' to get weights for each of the neurons
        z = np.dot(w, a_in) + b[j]                      #get wx+b
        a_out[j] = tc.sigmoid(tc.tensor(z))             #sigmoid(wx+b)  for each neuron
    return a_out 

def mySequential(x, W1, b1, W2, b2):        #instead of calling each layer separately, you can sequentially execute them with proper inputs for each layer
    a1 = myLinear(x,  W1, b1)
    a2 = myLinear(a1, W2, b2)
    return a2

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def myPredict(X, W1, b1, W2, b2):
    m = X.shape[0]                          #number of training examples
    p = np.zeros((m,1))                     #store probabilities
    for i in range(m):
        p[i,0] = mySequential(X[i], W1, b1, W2, b2)     #go through each example and get the probability for it
    return p

layer_norm = nn.LayerNorm([2,2], eps=1e-05, elementwise_affine=True)
X_tst = np.array([
    [157.,13.9],  # postive example
    [205.32,12.69]])   # negative example
X_tstn = layer_norm(tc.tensor(X_tst).float())  # remember to normalize
X_tstn_np = X_tstn.detach().numpy()           #convert normalized inputs back to numpy

predictions = myPredict(X_tstn_np, W1_tmp, b1_tmp, W2_tmp, b2_tmp)


yhat = np.zeros_like(predictions)
yhat = (predictions >= 0.5).astype(int)
print(f"\npredictions = \n{predictions[0]},{predictions[1]}")
print(f"decisions = \n{yhat}")
