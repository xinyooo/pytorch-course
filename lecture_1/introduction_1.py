import torch
import numpy as np

print(torch.__version__)

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

a = torch.ones(5)
b = a.numpy()
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(b)