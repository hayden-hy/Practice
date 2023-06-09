import torch
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


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
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])  # [[0,1],[1,2]] ; y_hat[0,1] = 0.1, y_hat[1,2]=0.5

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
    # sum{ -p(i)log(q(i))}
    # 因为结果往往是独热编码，所以除了选中的输出项为1，其余项都是0，因此只需要计算 -log()

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2) # correct, total
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n

    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def trian_epoch_ch3(net,train_iter,loss,updater):

    if isinstance(net,torch.nn.Module):
        net.eval()
    # total loss, total accuracy, num of sample
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean.backword()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())

    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = trian_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss,train_acc = train_metrics
    assert train_loss<0.5,train_loss
    assert train_acc<=1 and train_acc>0.7,train_acc
    assert  test_acc <=1 and test_acc>0.7,test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)

num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)

# predict

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net,test_iter)
d2l.plt.show()