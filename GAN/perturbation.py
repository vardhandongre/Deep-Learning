#------------------------------ Perturbation -----------------------------#
#                                                                         #                #
# Created by: Vardhan Dongre (vdongre2)                                   #                                          #
#-------------------------------------------------------------------------#

import torch
import torchvision
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.autograd as autograd
from torch.autograd import Variable
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

# Device configuration
torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Data
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)








def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(128, 1)
    alpha = alpha.expand(128, int(real_data.nelement()/128)).contiguous()
    alpha = alpha.view(128, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(128, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class Discriminator(nn.Module):
    def __init__ (self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,196,3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        self.ln1 = nn.LayerNorm((196,32,32))
        self.ln2 = nn.LayerNorm((196,16,16))
        self.ln3 = nn.LayerNorm((196,16,16))
        self.ln4 = nn.LayerNorm((196,8,8))
        self.ln5 = nn.LayerNorm((196,8,8))
        self.ln6 = nn.LayerNorm((196,8,8))
        self.ln7 = nn.LayerNorm((196,8,8))
        self.ln8 = nn.LayerNorm((196,4,4))
        

        
    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        x = F.leaky_relu(self.ln4(self.conv4(x)))
        x = F.leaky_relu(self.ln5(self.conv5(x)))
        x = F.leaky_relu(self.ln6(self.conv6(x)))
        x = F.leaky_relu(self.ln7(self.conv7(x)))
        x = F.leaky_relu(self.ln8(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196)
        fc1 = self.fc1(x)
        fc10 = self.fc10(x)
        return fc1, fc10
    
    
class Generator(nn.Module):
    def __init__ (self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.conv1 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(196,3,3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm1d(196*4*4)
        self.bn1 = nn.BatchNorm2d(196)
        self.bn2 = nn.BatchNorm2d(196)
        self.bn3 = nn.BatchNorm2d(196)
        self.bn4 = nn.BatchNorm2d(196)
        self.bn5 = nn.BatchNorm2d(196)
        self.bn6 = nn.BatchNorm2d(196)
        self.bn7 = nn.BatchNorm2d(196)
        
        

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1d(x)
        x = x.view(-1, 196, 4, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        x = torch.tanh(x)
        return x 




model = torch.load('cifar10.model')
model.cuda()
model.eval()

print("Perturbing...")

batch_idx, (X_batch, Y_batch) = testloader.next()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()



## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)




_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)


## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)



# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)


























