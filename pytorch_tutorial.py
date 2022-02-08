import torch

x = torch.rand(2,2)
y = torch.rand(2,2)

print(x)
print(y)
y = y.add_(x)
print(y)