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

# Loss function (criterion). Measures how wrong our prediction is compared to the real value.
loss_func = nn.CrossEntropyLoss()
# Optimizer is the judge. It detects how wrong the model is and corrects the values respectively.
# SGD = Stochastic Gradient Descent
# 1e-3 = 0.001
optimizer = torch.optim.SGD(model.parameters(),lr= 1e-3)

def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    # Sets the model into training mode
    model.train()
    # Training loop. Training 60000 images one batch at a time.
    # batch = batch number (0,1,2...), X = the batch of images (a tensor of shape [64,1,28,28]), y = batch of correct labels(a tensor of shape 64)
    # x,y in enumerate = x is the indexes in the dataloader and y is the corrresponding labels.
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # Prediction error
        # What is done here is basically run x, index = 0 for example through the model and store the output at pred.
        pred = model(X)
        # Then compare it to the label(y) related to out prediction and store the loss amount.
        loss = loss_func(pred, y)

        # Backpropagation
        # This function automatically calculates the gradient (the blame) for each parameter. It calculates how much each weight and bias contributed to the final loss.
        loss.backward()
        # This function uses the losses calculated by the loss.backward() and changes the parameters slightly according to the learning rate.
        optimizer.step()
        # Resets all of the gradients before the next step.
        optimizer.zero_grad()

        # Printing status update each 100 batches.
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# Here we are only testing (checking) the model, thus there is no optimizer and no updating weights.
def test(dataloader, model, loss_func):
    
    # Getting size to calculate overall accuracy. (8500/10000 for example)
    size = len(dataloader.dataset)
    # Getting the size of the batches to calculate average loss per batch.
    num_batches = len(dataloader)
    # Setting the model from training to evaluating mode.
    model.eval()
    # Initializing variables. test_loss stores loss from every single batch and correct stores every single prediction.
    test_loss, correct = 0,0

    # Use with keyword when connecting databases or handling files.
    # We are using no_grad() because since we are not training here, we are don't have to apply any backpropagation. We are just testing and seeing the results
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            # Calculates the loss for this batch and adding it's .item() value to test_loss.
            # We are using .item() to get the number out of the tensor thus prevent memory from building up.
            test_loss += loss_func(pred,y).item()
            # This line counts how many of them did we get correct.
            # Pred has a shape like [64,10]. pred.argmax(1) finds the index with the highest score (index from 0 to 9) from the batch and takes it as the final guess.
            # ==y operation compares the tensor of guesses which looks like [7,2,9] for example. If the ture y label is [7,2,1] it returns [True, True, False]
            # .type(torch.float) converts the true and false values into 1.0 and 0.0
            # .sum() adds up the 1.0 values and .item() converts these into plain python values like 2.
            # Lastly it adds this batches correctness value to the total.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # To calculate average loss per batch
    test_loss /= num_batches
    # To calculate the accuracy percentage
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Applying the created functions.
epochs = 5
for t in range(epochs):
    print(f"Epoch{t+1}\n--------------------------------")
    train(train_dataloader,model,loss_func,optimizer)
    test(test_dataloader,model, loss_func)
    print("Done!") 

# Saving the trained model
torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch model state to model.pth")

# # Loading the saved model
# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth",weights_only=True))

# # Use the model to make predictions
# classes = [
#         "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     x = x.to(device)
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f"Predicted: {predicted}, Actual: {actual}")
