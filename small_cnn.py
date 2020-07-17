import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms,utils,datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np


root_dir = 'data/2dconv'
train_dir = os.path.join(root_dir,'train')
validation_dir = os.path.join(root_dir,'validation')

train_kickflip_dir = os.path.join(train_dir,'kickflip')
train_ollie_dir = os.path.join(train_dir,'ollie')

# define the image preprocessing
# Data augmentation and normalization for training
# Just normalization for validation
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(155),
        transforms.CenterCrop(150),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(150),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the dataset
data_dir = 'data/2dconv'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          image_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

#map indexes to classes
idx_to_class = {v:k for k,v in image_datasets['train'].class_to_idx.items()}

print(image_datasets['train'],"\n\nClasses of the dataset",image_datasets['train'].classes)
print('index to class:',idx_to_class)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = self.convolution(c_in=3, c_out=16,kernel_size = 3, stride = 1)
        self.block2 = self.convolution(c_in=16,c_out=32,kernel_size=3,stride = 1)
        self.block3 = self.convolution(c_in = 32,c_out =64, kernel_size=3,stride=1)
        self.linear = self.linear_layers()
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1,64*17*17)
        return self.linear(x)

    def convolution(self,c_in,c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels= c_in,out_channels=c_out,**kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        return seq_block

    def linear_layers(self):
        linear_block = nn.Sequential(
            nn.Linear(in_features = 17*17*64,out_features=512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        return linear_block

if __name__ == "__main__":
    num_epochs = 80
    model = Model()
    if torch.cuda.is_available():
        model.to('cuda')
    optimizer = optim.RMSprop(params = model.parameters(),lr=0.001)
    for epoch in range(num_epochs):
        for imgs,classes in dataloaders['train']:
            imgs,classes = imgs.float(),classes.float()
            if torch.cuda.is_available():
                imgs,classes = imgs.to('cuda'),classes.to('cuda')
            output = model(imgs)
            loss   = F.binary_cross_entropy(output,classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, preds = torch.max(output,1)
                accuracy = torch.sum(preds == classes)
                print('Epoch {} \tLoss {:.4f} \tAcc: {}'
                      .format(epoch+1,loss.item(),accuracy.item()))

