import torch
from torch.autograd import Variable


#创建一个变量Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

#变量运算
y = x + 2
print(y)

#y作为一个运算的结果被创建，所以它有grad_fn
print(y.grad_fn)

#y更多运算
z = y**2*3
out = z.mean()
print(z, out)

#梯度gradients
out.backward() #out.backward() 等价于 out.backward(torch.Tensor[1.0]))
print(x.grad)

#更多案例by autograd
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
