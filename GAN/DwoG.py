#-------------------- Discriminator for CIFAR10 --------------------------#
#                                                                         #                #
# Created by: Vardhan Dongre                                              #                                          #
#-------------------------------------------------------------------------#

# Training a Generative Adversarial Network on CIFAR10 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.autograd as autograd
from torch.autograd import Variable
import time

# Device configuration
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 100
classes = 10
batch_size = 196
learning_rate = 0.0001

## Training Discriminator without Generator ##
# Prepare the data.
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


# Define the discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm((196, 32, 32))
        self.layer_norm2 = nn.LayerNorm((196, 16, 16))
        self.layer_norm3 = nn.LayerNorm((196, 16, 16))
        self.layer_norm4 = nn.LayerNorm((196, 8, 8))
        self.layer_norm5 = nn.LayerNorm((196, 8, 8))
        self.layer_norm6 = nn.LayerNorm((196, 8, 8))
        self.layer_norm7 = nn.LayerNorm((196, 8, 8))
        self.layer_norm8 = nn.LayerNorm((196, 4, 4))
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, classes)


    def forward(self, x):

        x = F.leaky_relu(self.layer_norm1(self.conv1(x)))
        x = F.leaky_relu(self.layer_norm2(self.conv2(x)))
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)))
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)))
        x = F.leaky_relu(self.layer_norm5(self.conv5(x)))
        x = F.leaky_relu(self.layer_norm6(self.conv6(x)))
        x = F.leaky_relu(self.layer_norm7(self.conv7(x)))
        x = F.leaky_relu(self.layer_norm8(self.conv8(x)))

        x = F.max_pool2d(x, kernel_size=4, padding=0, stride=4)

        x = x.view(x.size(0), -1)
        out1 = self.fc1(x)
        out10 = self.fc10(x)

        return (out1, out10)


# First Train the discriminator without generator 

model = discriminator()
model.to(device)

# Defining the loss and optimizer (Cross-Entropy and ADAM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Preparing to train......")

for epoch in range(100):

    
    total_correct = 0
    total = 0
    
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        
        if(Y_train_batch.shape[0] < 128):
            continue


        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)
        
        
        _, output = model(X_train_batch)
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        
        
        # training accuracy
        _, predicted = torch.max(output, 1)
        total += Y_train_batch.size(0)
        total_correct += (predicted == Y_train_batch).sum().item()
        
    print("Training Accuracy at epoch {}: {}".format(epoch, 100*(total_correct/total)))

    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'params_cifar10.ckpt') 
        torch.save(model, 'cifar10.model')
        

print("Finished Training!")

        
 

print("Preparing to Test.....")  
     
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print("Final Test Accuracy is {}".format(100*(correct/total)))





