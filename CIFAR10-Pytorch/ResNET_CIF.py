from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.BN1 = nn.Batchnorm2d(planes)
		self.BN2 = nn.Batchnorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or inplanes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias = False),
				nn.Batchnorm2d(self.expansion*planes)
				)

	def forward(self, x):
		out = self.conv1(x)
		out = self.BN1(x)
		out = F.relu(x)
		out = self.conv2(x)
		out = self.BN2(x)
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, basic_block, num_blocks, num_classes = 100):
		super(ResNet, self).__init__()
		channels = [32,64,128,256]

		self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.BN1 = nn.Batchnorm2d(32)
		self.drop = nn.Dropout2d(0.25)
		self.basic1 = self._add_layer(basic_block, channels[0], num_blocks[0], stride=1)
		self.basic2 = self._add_layer(basic_block, channels[1], num_blocks[1], stride=2)
		self.basic3 = self._add_layer(basic_block, channels[2], num_blocks[1], stride=2)
		self.basic4 = self._add_layer(basic_block, channels[3], num_blocks[0], stride=2)
		self.fc = nn.Linear(256*2*2*block.expansion, num_classes)

	def _add_layer(self, basic_block, planes, num_blocks, stride):
		stack = [stride] + [1]*(num_blocks-1)
		layer = []
		for i in stack:
			layer.append(basic_block(self.inplanes, planes, stride))
			self.inplanes = planes*block.expansion
		return nn.Sequential(*layer)


	def forward(self, x):
		out = self.conv1(x)
		out = self.BN1(out)
		out = F.relu(out)
		out = self.drop(out)
		out = self.basic1(out)
		out = self.basic2(out)
		out = self.basic3(out)
		out = self.basic4(out)
		out = F.max_pool2d(out, (2,2))
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return(out)


	def ResNET():
		return ResNet(BasicBlock, [2,4])




parser = argparse.ArgumentParser(description='Training on CIFAR100 using Pytorch')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 
start_epoch = 0  

# Preparing Data

print('Data Augmentation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# For trainning data
trainset = torchvision.datasets.CIFAR100(root=’~/scratch/’, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
# For testing data
testset = torchvision.datasets.CIFAR100(root=’~/scratch/’, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


# ResNET Model
model = ResNET()
model = model.to(device)
if deivce == 'cuda':
	model = torch.nn.DataParallel(model)
	cudnn.benchmark = True # Finds the best algorithm for the given config to use for the hardware (improves runtime) Not suitable if input size changes with each iteration

# checkpoint

if args.resume:
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']


criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=0)


def train(epoch):
	model.train()

	print("\n--- Epoch : %2d ---" % epoch)
    
    train_loss = 0
    correct = 0
    total_correct = 0
    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Training Loss: %.4f, Train Acc: %.4f ' % (loss.item(), 100.*correct/total_correct))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total_correct = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test Loss: %.4f, Test Acc: %.4f ' % (loss.item(), 100.*correct/total_correct))

    # Save checkpoint.
    acc = 100.*correct/total_correct
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+60):
    train(epoch)
    test(epoch)























