# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:46:56 2022

@author: KFS
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import torch.nn.functional as F
import torchattacks
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GTSRBDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_gtsrb_train_data(train_dataset_path, transform):
    images = []
    labels = []
    
    for root, _, files in os.walk(train_dataset_path):
        for file in files:
            if file.endswith('.ppm') or file.endswith('.png'):
                label = int(os.path.basename(root)) # Folder name as label
                img_path = os.path.join(root, file)
                images.append(img_path)
                labels.append(label)
    
    train_dataset = GTSRBDataset(images, labels, transform=transform)
    
    return train_dataset

# Load test dataset
def load_gtsrb_test_data(test_dataset_path, test_labels_file, transform):
    test_df = pd.read_csv(test_labels_file, delimiter=';')
    test_images = [os.path.join(test_dataset_path, img) for img in test_df['Filename']]
    test_labels = test_df['ClassId'].tolist()
    
    test_dataset = GTSRBDataset(test_images, test_labels, transform=transform)
    
    return test_dataset



### resnet50
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock_cbam(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_cbam, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=43):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
class ResNet_STN(nn.Module):

    def __init__(self, block, layers, num_classes=43):
        self.inplanes = 64
        super(ResNet_STN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1536),
            nn.ReLU(True),
            nn.Linear(1536, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    def stn_2(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def stn_3(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stn(x) + self.stn_2(x) + self.stn_3(x)
        # x = SpatialTransformer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_STN3(nn.Module):

    def __init__(self, block, layers, num_classes=43):
        self.inplanes = 64
        super(ResNet_STN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1536),
            nn.ReLU(True),
            nn.Linear(1536, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    def stn_2(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def stn_3(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stn(x) + self.stn_2(x) + self.stn_3(x)
        # x = SpatialTransformer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


#FGSM
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

batch_size = 256

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)), # Resize images to 32x32
    transforms.ToTensor(), # Convert images to PyTorch tensors
])
# Image preprocessing modules
train_dataset_path = 'C:\my_project_files\codes\codes\GTSRB\Final_Training\Images' # Set this to your training set folder
test_dataset_path = 'C:\my_project_files\codes\codes\GTSRB\Final_Test\Images' # Set this to your test images folder
test_labels_file = 'C:\my_project_files\codes\codes\GTSRB\GT-final_test.csv' # Set this to your test.csv file

train_dataset = load_gtsrb_train_data(train_dataset_path, transform)
test_dataset = load_gtsrb_test_data(test_dataset_path, test_labels_file, transform)
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# model_org = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
# model_STN = ResNet_STN(BasicBlock, [2, 2, 2, 2]).to(device)
# model_cbam = ResNet(BasicBlock_cbam, [2, 2, 2, 2]).to(device)
# model_STN_cbam = ResNet_STN(BasicBlock_cbam, [2, 2, 2, 2]).to(device)

model_org = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
model_STN = ResNet_STN(BasicBlock, [2, 2, 2, 2]).to(device)
model_cbam = ResNet(BasicBlock_cbam, [2, 2, 2, 2]).to(device)
model_STN_cbam = ResNet_STN(BasicBlock_cbam, [2, 2, 2, 2]).to(device)




###
num_epochs = 100
learning_rate = 0.001
total_step = len(train_loader)
curr_lr = learning_rate
# epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.1]
epsilons = [0, 0.03] #[8/255]
# epsilons = [0, 8/255]


results = []


####
model = model_org
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0, False)

for epsilon in epsilons:    
    test(epsilon, True, True,'ResNet-18 original, attack',results)

model = model_STN_cbam
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0, False)
for epsilon in epsilons:    
    test(epsilon, True, True,'ResNet-18-STN-CBAM original, attack',results)
   
model = model_org
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0.03, True)

for epsilon in epsilons:    
    test(epsilon, True, True,'ResNet-18 adversarial, attack',results)

model = model_STN_cbam
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
train(num_epochs,learning_rate, 0.03, True)
for epsilon in epsilons:    
    test(epsilon, True, True,'ResNet-18-STN-CBAM adversarial, attack',results)
   

  
    
    