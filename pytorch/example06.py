#(6) example06
'''
This is example about simple pytorch deep convolutional neural network(Using Residual Block)
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

epoch = 100
batch_size = 64

#######################
## preparing dataset ##
#######################
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root = './.data/', train = True, download = True, transform = transform)
testset= datasets.CIFAR10(root = './.data/', train = False, download = True, transform = transform)

train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size)
test_loader = data.DataLoader(dataset = testset, batch_size = batch_size)

'''
##check dataset
dataiter = iter(train_loader)
images, labels = next(dataiter)
img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize = (10,7))
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()
'''
########################
## defining  Residual ##
########################
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output

########################
## defining CNN model ##
########################
class DeepCNN(nn.Module):
    def __init__(self, class_num=10):
        super(DeepCNN,self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.fc = nn.Linear(64, class_num)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = F.max_pool2d(output, 8)
        output = out.view(out.size(0), -1)
        output = self.fc(output)
        return output


#######################
## training for data ##
#######################
model = DeepCNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train epoch : {} [{}/{} {:.0f}%]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, epoch + 1):
    scheduler.step()
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss : {:4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))

#######################
## saving the weight ##
#######################
## save
torch.save(model.state_dict(), 'model.pt')

## load
new_model = CNN()
new_model.load_state_dict(torch.load('model.pt'))
new_model.eval()
