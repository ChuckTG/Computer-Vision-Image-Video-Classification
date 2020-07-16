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
        transforms.RandomResizedCrop(150),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/2dconv'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          image_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the dataset
skate_dataset = datasets.ImageFolder(root = train_dir,
                                     transform = image_transforms['train'])


#map indexes to classes
idx_to_class = {v:k for k,v in skate_dataset.class_to_idx.items()}

print(skate_dataset,"\n\nClasses of the dataset",skate_dataset.classes)
print('index to class:',idx_to_class)

