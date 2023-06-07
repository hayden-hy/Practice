import numpy as np
import torch
import torch.utils.data as data
from d2l import torch as d2l

true_w = torch.tensor([2,-3.4])
true_b  = 4.2

features,labels = d2l.synthetic_data(true_w,true_b,1000)

# data iterator
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)

    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)

# define model
from torch import nn
net = nn.Sequential(nn.Linear(2,1))

# parameter initialization
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# define loss func
loss = nn.MSELoss()

# define optimizer alg
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

# train
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
