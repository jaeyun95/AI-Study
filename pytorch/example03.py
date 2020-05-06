#(3) example03
'''
This is example about simple pytorch neural network
'''
import torch
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch.nn.functional as F

#######################
## preparing dataset ##
#######################
x_train, y_train = make_blobs(n_samples = 80, n_features = 2, centers=[[1,1],[-1,-1],[1,-1],[-1,1]],shuffle = True, cluster_std = 0.3)
x_test, y_test = make_blobs(n_samples = 20, n_features = 2, centers=[[1,1],[-1,-1],[1,-1],[-1,1]],shuffle = True, cluster_std = 0.3)

## visual function
def vis_data(x, y = None, c = 'r'):
    if y is None:
        y = [None] * len(x)
    for x_, y_ in zip(x,y):
        if y_ is None:
            plt.plot(x_[0], x_[1], '*',markerfacecolor = 'none',markeredgecolor = c)
        elif y_ == 0:
            plt.plot(x_[0], x_[1], c+'o')
        elif y_ == 1:
            plt.plot(x_[0], x_[1], c+'+')
        elif y_ == 2:
            plt.plot(x_[0], x_[1], c+'x')
        elif y_ == 3:
            plt.plot(x_[0], x_[1], c+'^')

#check data graph
'''
plt.figure()
vis_data(x_train, y_train, c = 'r')
plt.show
'''

## convert numpy format to tensor format
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

#######################
## defining NN class ##
#######################
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor):
        linear1 = self.linear_1(input_tensor)
        relu = self.relu(linear1)
        linear2 = self.linear_2(relu)
        output = self.sigmoid(linear2)
        return output

#######################
## training for data ##
#######################
model = NeuralNetwork(2, 5)
learning_rate = 0.03
criterion = torch.nn.BCELoss()
epochs = 10000
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

model.eval()
test_loss_before = criterion(model(x_test).squeeze(), y_test)
print('Before Training, test loss is {}'.format(test_loss_before.item()))

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_output = model(x_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    if epoch%100 == 0:
        print('Train loss at {} is {}'.format(epoch, train_loss.item()))
    train_loss.backward()
    optimizer.step()
model.eval()
test_loss_before = criterion(torch.squeeze(model(x_test)),y_test)
print('After Training, test loss is {}'.format(test_loss_before.item()))