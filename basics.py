# While working with data we generally use two primitives. torch.utils.data.DataLoader and torch.utils.data.Dataset.
# Dataset stores the samples and their labels, DataLoader wraps an iterable around the Dataset to allow an easy access.

import torch
from torch.utils.data import DataLoader
# There are lots of domain spesific libraries in torch like TorchText, TorchVision or TorchAudio which include datasets.
from torchvision import datasets
# torchvision.datasets module includes datasets for many real-world vision data like CIFAR, COCO etc.
from torchvision.transforms import ToTensor

# In this practice, we will use the FashionMNIST dataset.

# Download the training data.
training_data = datasets.FashionMNIST(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download the test data.
test_data = datasets.FashionMNIST(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Creating dataloaders so that it wraps an iterable around the dataset and we can work on it easily.

batch_size = 64 
# This means each element in the dataloader iterable will return a batch of 64 features and labels.

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break