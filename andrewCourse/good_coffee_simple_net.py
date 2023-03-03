import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as tc

X_np = np.array([[185.32,12.69],
 [259.92,11.87],
 [231.01,14.41],
 [175.37,11.72],
 [187.12,14.13],
 [225.91,12.1 ],
 [208.41,14.18],
 [207.08,14.03],
 [280.6,14.23],
 [202.81,12.25]])
Y_np = np.array([[1.,0.,0.,0.,1.,1.,0.,0.,0.,1.]])      #DATA

X_tensor = tc.from_numpy(X_np).float()      #Convert input to tensor so you can normalize

layer_norm = nn.LayerNorm([10,2], eps=1e-05, elementwise_affine=True)
norm_X_tensor = layer_norm(X_tensor)
norm_X_numpy = norm_X_tensor.detach().numpy()           #convert normalized inputs back to numpy

def myLinear(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)                         
    for j in range(units):               
        w = W[:,j]                                      #         
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = tc.sigmoid(tc.tensor(z))               
    return a_out 

def mySequential(x, W1, b1, W2, b2):
    a1 = myLinear(x,  W1, b1)
    a2 = myLinear(a1, W2, b2)
    return a2

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

def myPredict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = mySequential(X[i], W1, b1, W2, b2)
    return p
