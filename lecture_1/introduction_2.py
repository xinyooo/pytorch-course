import torch

x = torch.randn(3, 4, requires_grad=True)
print(x)

x = torch.randn(3, 4)
x.requires_grad = True
print(x)

b = torch.randn(3, 4, requires_grad=True)
t = x + b
y = t.sum()
print(y)

y.backward()

print(b.grad)

print(x.requires_grad, b.requires_grad, t.requires_grad)

x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = w * x
z = y + b
print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad)

print(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf)

z.backward(retain_graph=True)
print(w.grad)

print(b.grad)