import torch
import time
import numpy as np

N = 4096

A = np.random.randn(N,N).astype(np.float32)
B = np.random.randn(N,N).astype(np.float32)

flop = N*N*2*N
print(f"{flop / 1e9:.2f} GFLOP")
st = time.monotonic()
C = A @ B
et = time.monotonic()
s = et - st
print(f"{flop/s * 1e-12:.2f} TFLOP/S")



# define a tensor
#A = torch.tensor(5., requires_grad=True)
#print("Tensor-A:", A)
  
# define a function using above defined
# tensor
#y = A**3
#print("x:", y)
#y.backward()
# print the gradient using .grad
#print("A.grad:", A.grad)