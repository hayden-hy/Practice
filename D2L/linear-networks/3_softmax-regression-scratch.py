import torch
from d2l import torch as d2l

num_inputs = 784
num_outputs = 10

W = torch.normal(0,0.01,size =(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

X = torch.arange(1,7,dtype=float).reshape(2,3)
print(X)
print(X.sum(0,keepdim=True))
print(X.sum(1,keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition

X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))

# model define
def net(X):
    return softmax(torch.matmul((X.reshape(-1,W.shape[0]),W))+b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])  # [[0,1],[1,2]] ; y_hat[0,1] = 0.1, y_hat[1,2]=0.5

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

# 因为结果往往是独热编码，所以除了选中的输出项为1，其余都是0，因此只需要计算 -log()

