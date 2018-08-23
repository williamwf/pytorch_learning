#=========训练神经网络步骤===========#
"""
1）定义神经网络，以及一些可以学习的参数
2）输入数据集迭代
3）对输出数据处理
4）计算loss
5）梯度反向传播
6）更新网络权重  weight = weight - learning_rate* gradient
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#1 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1层输入，6个输出，5*5卷积
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #2*2最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#学习到的参数可以被net.parameters()返回
params = list(net.parameters())
print(len(params))
print(params[0].size())


#前向计算的输入和输出都是autograd.Variable
#lenet的输入尺寸是32*32，为了在MNIST数据集上使用这个网络，请把图像大小转变为32*32。
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

#将梯度缓冲区归零，然后使用随机梯度值进行反向传播。
net.zero_grad()
out.backward(torch.randn(1,10))

#2- 损失函数
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  #保持和输出一样的维度
criterion = nn.MSELoss()
print("target:", target)

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  #MSELoss
print(loss.grad_fn.next_functions[0][0])  #Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  #relu


#3- 反向传播
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#4 更新权重
#SGD

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

#不同更新法则
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # update
