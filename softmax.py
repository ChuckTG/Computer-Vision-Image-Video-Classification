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
import time
import copy
from small_cnn import Model
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
        transforms.Resize(size = (150,150)),
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

batch_size = 100
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

#map indexes to classes
idx_to_class = {v:k for k,v in image_datasets['train'].class_to_idx.items()}

print(image_datasets['train'],"\n\nClasses of the dataset",image_datasets['train'].classes)
print('index to class:',idx_to_class)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__name__":

    convnet = Model()
    convnet = convnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(params = convnet.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    num_epochs = 10

    #train the model and return best model
    model = train_model(convnet,criterion,optimizer,
                        scheduler,num_epochs=num_epochs)


