# While working with data we generally use two primitives. torch.utils.data.DataLoader and torch.utils.data.Dataset.
# Dataset stores the samples and their labels, DataLoader wraps an iterable around the Dataset to allow an easy access.

import torch
from torch.utils.data import DataLoader
# There are lots of domain spesific libraries in torch like TorchText, TorchVision or TorchAudio which include datasets.
from torchvision import datasets
# torchvision.datasets module includes datasets for many real-world vision data like CIFAR, COCO etc.
from torchvision.transforms import ToTensor
import torch.nn as nn

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

# Use GPU if available
device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  in newer pytorch versions
print(f"Using {device}")

# Creating the neural network
class NeuralNetwork(nn.Module):
    # The constructor
    def __init__(self):
        super().__init__()
        # Flatten 2D pixels (28*28 for example) into 1D (784)
        self.flatten = nn.Flatten()
        # Self keyword causes the created layers to be stored inside here, without it they would be just temporary variables.
        self.linear_relu_stack = nn.Sequential(
            # First layer, transform 784 pixels to 512 with weights and biases
            nn.Linear(28*28, 512),
            # If the data is positive leave it as it is. If not, make them 0. Thus, add non-linearity to model and allow it to handle more complex operations.
            nn.ReLU(),
            # Second layer
            nn.Linear(512,512),
            nn.ReLU(),
            # Third layer
            nn.Linear(512,10)
        )

    def forward(self, x):
        # This code translates to "Go find the flatten layer I created and pass the data x through it"
        x = self.flatten(x)
        # This code translated to "Now go find the linear_relu_stack and pass the data x through it"
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Optimizing the parameters

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr= 1e-3)

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # Prediction error
        pred = model(X)
        loss = loss_func(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred,y).item()
            correct += (pred.argmax(1)== y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch{t+1}\n--------------------------------")
    train(train_dataloader,model,loss_func,optimizer)
    test(test_dataloader,model, loss_func)
    print("Done!") 
