# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 00:46:44 2025

@author: hadi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init
import torchattacks

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./MNIST_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./MNIST_data/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes=512, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

######################################################################
# spatial transformer networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # self.ca = ChannelAttention(in_planes=20)
        # self.sa = SpatialAttention()
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
  
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

######################################################################
# spatial transformer networks
class Net_STN_CBAM(nn.Module):
    def __init__(self):
        super(Net_STN_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.ca = ChannelAttention(in_planes=20)
        self.sa = SpatialAttention()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    def stn_2(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def stn_3(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x) + self.stn_2(x) + self.stn_3(x)
        # x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        attention_values = self.ca(x)
        x = attention_values * x
        x = self.sa(x) * x

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


######################################################################

def fgsm_attack(input,epsilon,data_grad):
  pert_out = input + epsilon*data_grad.sign()
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out


# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
def train(num_epochs,learning_rate,epsilon,attack_state):
    curr_lr = learning_rate

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            images.requires_grad = True
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # FGSM
            if attack_state:
                model.zero_grad()
                # Collect datagrad
                data_grad = images.grad.data
                # epsilon = 0.6
                # Call FGSM Attack
                #attack = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=5, random_start=False)
                attack = torchattacks.FGSM(model, eps=epsilon)
                perturbed_data = attack(images, labels)
                
                
                # perturbed_data = fgsm_attack(images, epsilon, data_grad)
                outputs = model(perturbed_data)
                loss = criterion(outputs, labels)
                loss.backward() 
                optimizer.step()
                  
                
        
                
        # Decay learning rate
        if (epoch + 1) % 5 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

# Test the model


def test(epsilon,attack_state,train_mode,net_type,results):

    model.eval()
    # with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)            
        images.requires_grad = True            
        outputs = model(images) 
        loss = criterion(outputs, labels)    
            # Backward and optimize
        model.zero_grad()            
        if images.grad is not None:
            images.grad.data.fill_(0)
            
        loss.backward()            
        data_grad = images.grad.data
        
        #attack = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=5, random_start=False)
        attack = torchattacks.FGSM(model, eps=epsilon)
        perturbed_data = attack(images, labels)
        
        
        # perturbed_data = fgsm_attack(images, epsilon, data_grad)
        if attack_state:
            outputs = model(perturbed_data)
        else:
            outputs = model(images)            
            
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()    
    print('Model:{}, Epsilon: {:.4f}, Accuracy: {}%, adversarial training: {}'.format(net_type,epsilon, 100 * correct / total, train_mode))
    results.append('\n Model:{}, Epsilon: {:.4f}, Accuracy: {}%, adversarial training: {}'.format(net_type,epsilon, 100 * correct / total, train_mode))
        # Save the model checkpoint
    file = open('test.txt', 'w')
    file.write(''.join(results))
    file.close()
    # torch.save(results,'test.txt' )
    
####


###
num_epochs = 10
learning_rate = 0.001
total_step = len(train_loader)
curr_lr = learning_rate
epsilons = [0, 0.2, 0.4, 0.6, 0.8]
# epsilons = [0, 0.03] #[8/255]
# epsilons = [0, 8/255]


results = []


####
model = Net()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0, False)

for epsilon in epsilons:    
    test(epsilon, True, True,'LeNet original, attack',results)

model = Net_STN_CBAM()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0, False)
for epsilon in epsilons:    
    test(epsilon, True, True,'LeNet-STN-CBAM original, attack',results)
   
for epsilon in epsilons:
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    train(num_epochs,learning_rate, epsilon, True)
    test(epsilon, True, True,'LeNet adversarial, attack',results)

    model = Net_STN_CBAM()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    train(num_epochs,learning_rate, epsilon, True)
    test(epsilon, True, True,'LeNet-STN-CBAM adversarial, attack',results)
   

  
    
    