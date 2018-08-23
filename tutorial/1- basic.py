from __future__ import print_function
import torch
import numpy as np


#1 Tensors
#初始化矩阵
x = torch.Tensor(5, 3)
print(x)
print(x.size())

#2 运算
y = torch.rand(5, 3)
print(x + y)
#print(torch.add(5, 3)
print(x[:-1])


#3 NumpyBridge
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
