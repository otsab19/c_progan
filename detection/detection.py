import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import torch.nn.functional as F

#for merged
MERGED_DATA_PATH = 'Merged data set'
#for original
DATA_PATH = 'Brain Tumor Data Set'
# Define the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ask if hyper tuning to run or not

user_input = input("Perform hyperparameter tuning? (yes/no): ").lower()


# Convolutional Block Definition
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, use_pooling=True, use_activation='relu'):
        super().__init__()
        activation = nn.ReLU() if use_activation == 'relu' else nn.LeakyReLU(0.1)
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            activation
        ]

        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(activation)

        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Network Definition
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            BasicConvBlock(3, 32, 2, use_pooling=False, use_activation='relu'),
            BasicConvBlock(32, 64, 2, use_pooling=False, use_activation='leaky_relu'),
            BasicConvBlock(64, 128, 2, use_pooling=True, use_activation='leaky_relu'),
            BasicConvBlock(128, 256, 2, use_pooling=True, use_activation='leaky_relu'),
            BasicConvBlock(256, 512, 3, use_pooling=True, use_activation='leaky_relu'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 2)  # Adjusted to have 2 output neurons
        )

    def forward(self, xb):
        return self.network(xb)


# Instantiate and send the model to device
network = Network().to(DEVICE)

# Define transformations and data loaders
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # resizes the image so it can be perfect for our model.
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  # Flips the image w.r.t horizontal axis
    transforms.RandomRotation(10),  # Rotates the image to a specified angle
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Performs actions like zooms, change shear angles.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
    transforms.ToTensor(),  # convert the image to tensor so that it can work with torch
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
data_path = os.path.join(os.getcwd(), MERGED_DATA_PATH)
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

print(dataset.class_to_idx)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

# Training the network
num_epochs = 35
for epoch in range(num_epochs):
    network.train()  # Ensure the network is in train mode
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # labels = labels.unsqueeze(1).float()  # Adjust labels to match output
        optimizer.zero_grad()
        outputs = network(images)
        # loss = criterion(outputs, labels)
        loss = F.cross_entropy(outputs, labels)  # Calculate Loss
        loss.backward()
        optimizer.step()

    # Print out loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the network
network.eval()  # Set the network to evaluation mode
true_labels = []
predicted_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = network(images)
        predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate and print the accuracy
accuracy = (np.array(true_labels) == np.array(predicted_labels)).mean() * 100
print(f'Accuracy of the model on the test images: {accuracy:.2f} %')

# Plotting the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
classes = ['No Tumor', 'Tumor']
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add the numbers inside the cells
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.show()


# for hyperparameters tuning
def objective(hyperparams):
    batch_size = int(hyperparams['batch_size'])
    lr = hyperparams['lr']
    optimizer_choice = hyperparams['optimizer']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Network().to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification

    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training loop
    for epoch in range(10):  # Reduced number of epochs for hyperparameter tuning
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # Use CrossEntropyLoss
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return {'loss': -accuracy, 'status': STATUS_OK}


if user_input == 'yes':
    # Define the space of hyperparameters to search
    space = {
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'optimizer': hp.choice('optimizer', ['Adam', 'SGD'])
    }

    # Running the optimizer
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
    print("Best hyperparameters:", best)
