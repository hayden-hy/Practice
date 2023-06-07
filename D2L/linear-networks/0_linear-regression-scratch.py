import random

import torch
from d2l import torch as d2l


# generate data
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # matmul(A,B)
    # 1: (A.row x A.col) x (B.col) = A.row if A.col==B.row and B = (1,col), namely (1,col)
    # 2: (A.row x A.col) x (B.row,B.col) = (A.row,B.col)  if A.col==B.row
    print(X.shape)
    print(w.reshape(-1, 1).shape)
    print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# plot scatter chart
d2l.set_figsize()
d2l.plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()


# data iterator
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# model parameter initialization
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# define model
def linreg(X, w, b):
    return torch.matmul(X, w) + b  # return X.row

# define loss func
def squared_loss(y_hat,y):
    return (y-y_hat.reshape(y.shape)) ** 2 / 2

# define optimizer algorithm
def sgd(params,lr,batch_size):
    with torch.no_grad(): # don't calculate grad
        for para in params:
            para -= lr*para.grad / batch_size
            para.grad.zero_()

# train
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10 # 越大精度越低
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()              # gradient accumulated
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')