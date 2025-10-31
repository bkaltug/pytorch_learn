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

