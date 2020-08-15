#(7) example07
'''
This is example about simple pytorch autoencoder
'''
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torch.utils import data

epoch = 10
batch_size = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#######################
## preparing dataset ##
#######################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

trainset = datasets.FashionMNIST(root = './.data/', train = True, download = True, transform = transform)
testset= datasets.FashionMNIST(root = './.data/', train = False, download = True, transform = transform)

train_loader = data.DataLoader(dataset = trainset, batch_size = batch_size)
test_loader = data.DataLoader(dataset = testset, batch_size = batch_size)


################################
## defining AutoEncoder model ##
################################
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encode_data = self.encoder(x)
        decode_data = self.decoder(encode_data)
        return encode_data, decode_data


#######################
## training for data ##
#######################
autoencoder = AutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()

view_data = trainset.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255.

def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)
        encode_data, decode_data = autoencoder(x)
        loss = criterion(decode_data, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for e in range(1, epoch+1):
    train(autoencoder, train_loader)
    test_x = view_data.to(DEVICE)
    _, decode_data = autoencoder(test_x)
	print("[EPOCH {}]".format(e))
	'''
    f, a = plt.subplots(2, 5, figsize=(5,2))
    
    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i],(28,28))
        a[0][i].imshow(img, cmap='gray')
    for i in range(5):
        img = np.reshape(view_data.to("cpu").data.numpy()[i],(28,28))
        a[1][i].imshow(img, cmap='gray')
    plt.show()
    '''

	
##################
## show data 3D ##
##################
view_data = trainset.data[:100].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor)/255.
test_x = view_data.to(DEVICE)
encode_data, _ = autoencoder(test_x)
encode_data = encode_data.to("cpu")
CLASS_LABEL = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)
X = encode_data.data[:,0].numpy()
Y = encode_data.data[:,1].numpy()
Z = encode_data.data[:,2].numpy()

label = trainset.targets[:100].numpy()

for x, y, z, s in zip(X, Y, Z, label):
    name = CLASS_LABEL[s]
    color = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, name, backgroundcolor=color)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
