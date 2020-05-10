#(4) example04
'''
This is example about simple pytorch deep neural network
'''
import torch
from torchvision import datasets, transforms, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#######################
## preparing dataset ##
#######################
transform = transforms.Compose([transforms.ToTensor()])

trainset= datasets.FashionMNIST(root = './.data/', train = True, download = True, transform = transform)
testset= datasets.FashionMNIST(root = './.data/', train = False, download = True, transform = transform)

batch_size = 16
train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size)
test_loader = data.DataLoader(dataset = testset, batch_size = batch_size)

##check dataset
'''
dataiter = iter(train_loader)
images, labels = next(dataiter)
img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize = (10,7))
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()
'''

EPOCHS = 30
BATCH_SIZE = 64

########################
## defining DNN model ##
########################
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,input_data):
        x = input_data.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#######################
## training for data ##
#######################
model = DeepNeuralNetwork().to(DEVICE)
optimizer = optimizer = optim.SGD(model.parameters(),lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1,EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))

#######################
## saving the weight ##
#######################
## save
torch.save(model.state_dict(), 'model.pt')

## load
new_model = DeepNeuralNetwork()
new_model.load_state_dict(torch.load('model.pt'))
new_model.eval()