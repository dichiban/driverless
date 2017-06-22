from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import data

class CNNet(nn.Module):

    def __init__(self):

        super(CNNet, self).__init__()


        #Convolutional layers

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=24,
                               kernel_size=(5, 5),
                               stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=24,
                               out_channels=36,
                               kernel_size=(5,5),
                               stride=(2,2))

        self.conv3 = nn.Conv2d(in_channels=36,
                               out_channels=48,
                               kernel_size=(5,5),
                               stride=(2,2))

        self.conv4 = nn.Conv2d(in_channels=48,
                               out_channels=64,
                               kernel_size=(3,3))

        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3,3))

        self.conv6 = nn.Conv2d(in_channels=64,
                               out_channels=1164,
                               kernel_size=(5, 5))


        #Fully-connected layers
        self.fc1 = nn.Linear(in_features=1164, out_features=100)

        self.fc2 = nn.Linear(in_features=100, out_features=50)

        self.fc3 = nn.Linear(in_features=50, out_features=10)

        self.fc4 = nn.Linear(in_features=10, out_features=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("conv1 ", x.size())
        x = F.relu(self.conv2(x))
        #print("conv2 ", x.size())
        x = F.relu(self.conv3(x))
        #print("conv3 ", x.size())
        x = F.relu(self.conv4(x))
        #print("conv4 ", x.size())
        x = F.relu(self.conv5(x))
        #print("conv5 ", x.size())
        x = F.relu(self.conv6(x))
        #print("conv6 ", x.size())

        x = x.view(-1, 1164)

        #print("ravel ", x.size())
        x = F.relu(self.fc1(x))
        #print("fc1 ", x.size())
        x = F.relu(self.fc2(x))
        #print("fc2 ", x.size())
        x = F.relu(self.fc3(x))
        #print("fc3 ", x.size())
        x = F.relu(self.fc4(x))
        #print("out ", x.size())

        return x


model = CNNet().cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# Load Training Data
x_train, x_valid, y_train, y_valid = data.load_data()

x_train = np.transpose(x_train, (0, 3, 1, 2))

data = torch.from_numpy(x_train).float()
x_valid = torch.from_numpy(x_valid).float()
target = torch.from_numpy(y_train).float()
y_valid = torch.from_numpy(y_valid).float()
data, target = data.cuda(), target.cuda()
data, target = Variable(data), Variable(target)

train = data_utils.TensorDataset(data, target)
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

def train(epoch):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if i % epoch == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i* len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.data[0]))


epoch = 100
for epoch in range(1, epoch + 1):
    train(epoch)

print('Finished Training')



