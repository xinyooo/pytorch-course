import torch
from torch import tensor

# 0:scalar
# 1:vector
# 2:matrix
# 3:n-dimensional tensor


# scaler
x = tensor(42.)
print(x)

print(x.dim())

# vector
v = tensor([1.5, -0.5, 3.0])
print(v)

print(v.dim())

print(v.size())

# matrix
M = tensor([[1., 2.], [3., 4.]])
print(M)

print(M.matmul(M))

print(tensor([1., 0.]).matmul(M))

print(M*M)

print(tensor([1., 2.]).matmul(M))