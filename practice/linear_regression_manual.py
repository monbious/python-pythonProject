import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

learning_rate = 0.01

x = torch.rand([500,1])
print(x)
y_true = x*3 + 0.8

w = torch.rand([1,1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

for i in range(200):
    y_predict = torch.matmul(x, w) + b
    loss = (y_true - y_predict).pow(2).mean()

    loss.backward()

    w.data = w.data - learning_rate*w.grad
    b.data = b.data - learning_rate*b.grad

    if i%50 == 0:
        print(f'w: {w.item()}, b: {b.item()}, loss: {loss.item():.6f}')

        if w.grad is not None:
            w.grad.data.zero_()
        if b.grad is not None:
            b.grad.data.zero_()


plt.figure(figsize=(20,8))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c='r')
plt.show()













