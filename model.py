# importing libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

# Checking for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(device)

# transforms
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Path for traning and test data
train_path = ""
test_path = ""


# Dataloader
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=100, shuffle=True

)

test_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=256, shuffle=True

)

# categories
root =pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Input shae=(256, 3, 150, 150)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        #Shape= (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        #Shape= (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        #Shape= (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # reduce the image size be factor 2
        # Shape= (256, 12, 75, 75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        #Shape= (256, 20, 75, 75)
        self.relu2=nn.ReLU()
        #Shape= (256, 20, 75, 75)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        #Shape= (256, 32, 75, 75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256, 32, 75, 75)
        self.relu3=nn.ReLU()
        #Shape= (256, 32, 75, 75)

        self.fc=nn.Linear(in_features=32*75*75, out_features=num_classes)

    # Feed forward function

    def forward(self, input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)

        output=self.pool(output)

        output=self.conv2(output)
        output=self.relu2(output)

        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        # Above output will be in matrix form, with shae(256, 32, 75, 75)

        output=output.view(-1, 32*75*75)

        output=self.fc(output)

        return output

model = ConvNet(num_classes=6).to(device)

# Optimizer and Loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs=10

# Calculating the size of traning and testing images
train_count=len(glob.glob(train_path+"/**/*.png"))
test_count=len(glob.glob(test_path+"/**/*.png"))

print(train_count, test_count)

# Model training and saving bestmodel

best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and traning on training data
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        train_accuracy+=torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        print("Epoch: {}/{} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f}".format(epoch+1, num_epochs, loss.item(), train_accuracy/(i+1))) 

        train_accuracy=train_accuracy/train_count
        train_loss=train_loss/train_count

    # Evaluation on testing data
    model.eval()


    test_accuracy=0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_accuracy+=torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        print("Epoch: {}/{} \tTesting Loss: {:.4f} \tTesting Accuracy: {:.4f}".format(epoch+1, num_epochs, loss.item(), test_accuracy/(i+1))) 

        test_accuracy=test_accuracy/test_count


    # Saving the best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        print("Saving the best model")
        best_accuracy=test_accuracy  