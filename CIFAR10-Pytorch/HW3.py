#-------------------- Deep NN using Pytorch for CIFAR10 ------------------#
#                                                                         #
# Implementation of Deep Neural Network for classifying CIFAR10 dataset.  #
# Target accuracy on Test Set was 80 -90%, This                           #
# implementation achieved 84% accuracy with the follwing hyper-           #
# parameters:                                                             #
# Epochs = 100, batch size = 100, lr = 0.001, lr schedule provided        #
# Model Architecture:                                                     #
# Conv -> RelU -> Conv -> RelU -> Maxpool -> Dropout -> Conv -> RelU ->   #
# Conv -> RelU -> Maxpool -> Dropout -> linear -> RelU -> linear ->       #
# dropout -> linear                                                       #  
# Trained on Google Colab with GPU as hardware accelerator                #
#                                                                         #
# Created by: Vardhan Dongre                                              #
#-------------------------------------------------------------------------#


import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

# Hyper parameters
num_epochs = 100
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Device 
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set seed
torch.manual_seed(1)


# Data Augmentation
print('Data Preparation and Augmentation......')


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR10 Data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


## Model Architecture
class Deepnet(nn.Module):
    def __init__(self):
        super(Deepnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(48, 96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(96, 192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(192, 256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8*8*256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.pool(x) 
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


model = Deepnet().to(device)



# Defining the loss and optimizer (Cross-Entropy and ADAM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


# Model Training
main_step = 0


def train(epoch):
    model.train()
    scheduler.step()

    print("\n ____Epoch: %2d ____" % epoch)

    steps = 50000//batch_size

    if(epoch > 6):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if(state['step'] >= 1024):
                    state['step'] = 1000
    optimizer.step()

    for step, (images, labels) in enumerate(train_loader, 1):
        global main_step
        main_step += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, step, steps, loss.item()))
        


def eval(epoch):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print("Accuracy : %.4f" % (total_correct/total), file=f)
    print("Accuracy : %.4f" % (total_correct/total))



with open('result.txt', 'w') as f:
    for epoch in range(1, num_epochs+1):
        train(epoch)
        eval(epoch)
f.close()



# Testing
model.eval() 
with torch.no_grad():
    total_correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_correct += (predicted == labels).sum().item()

print('Accuracy of the model: {} %'.format(100 * total_correct / total))


torch.save(model.state_dict(), 'model_cifar10_batch.pkl')