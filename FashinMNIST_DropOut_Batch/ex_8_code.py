# Eliad Arzuan 206482622
from __future__ import print_function
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Hidden layer size
first_hidden = 100
second_hidden = 50
num_classes = 10

# Out learning rate
eta = 0.01
batch_size = 48
val_percent = 0.2

image_size = 28 * 28



# Loading our fashion mnist
def loadMnist() :
    #Transform the image into normalized tensor
    normalizedTransofmrs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    fashionSet = datasets.FashionMNIST('./data', train=True, download=True,
                   transform=normalizedTransofmrs)

    num_train = len(fashionSet)
    indices = list(range(num_train))
    split = int(val_percent * num_train)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]
    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)

    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=fashionSet,
                                               batch_size=batch_size, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset=fashionSet,
                                                    batch_size=batch_size, sampler=validation_sampler)


    # Loading the train set
    test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, transform=normalizedTransofmrs),
    batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader


# Loading our mnist
train_loader, validation_loader, test_loader = loadMnist()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.fc3 = nn.Linear(second_hidden, num_classes)

    def forward(self, x):
        # Reshape x - -1 - because we want to b sure
        x = x.view(-1, self.image_size)
        # Hidden layers will be activated by relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
 #       x = F.relu(self.fc3(x))
        return F.log_softmax(x, dim = 1)

#def DropoutNet
class DropoutNet(nn.Module):
    def __init__(self, prob = 0.35):
        super(DropoutNet, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.fc3 = nn.Linear(second_hidden, num_classes)
        # Adding dropout
        self.dropout = nn.Dropout(prob)

    def forward(self, x):
        # Reshape x - -1 - because we want to b sure
        x = x.view(-1, self.image_size)
        # Hidden layers will be activated by relu
        # We will do dropout after each output in hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #       x = F.relu(self.fc3(x))
        return F.log_softmax(x, dim=1)


class BatchNet(nn.Module):
    def __init__(self, size = 200):
        super(BatchNet, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, first_hidden)
        self.batch_norm1 = nn.BatchNorm1d(first_hidden)

        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.batch_norm2 = nn.BatchNorm1d(second_hidden)

        self.fc3 = nn.Linear(second_hidden, num_classes)
        # Adding batchnorm
    def forward(self, x):
        # Reshape x - -1 - because we want to b sure
        x = x.view(-1, self.image_size)
        # Hidden layers will be activated by relu
        # We will do dropout after each output in hidden layer
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
# Training the model
def train(epoch, model, optimizer, train_loader):
    model.train()
    # Moving on all
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()




def avgLossCheck(model, loader, name):
    model.eval()
    loss = 0.0
    correct = 0.0
    for data, target in loader:
        output = model(data)
        loss += float(F.nll_loss(output, target, size_average=False).item())  # sum up batch loss
        # get the index of the max log-probability - argmax
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= float(len(loader) * batch_size)
    print('\n',name,' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader) * batch_size,
        100. * correct / (len(loader) * batch_size)))
    return loss
# Testing the model
def test(test_loader, model):
    avgLossCheck(model, test_loader, "Test")

def writePred(model):
    model.eval()
    # Write into file
    with open('test.pred', 'w') as test_y:
        for data, target in test_loader:
            output = model(data)
            # get the index of the max log-probability - argmax
            pred = output.data.max(1, keepdim=True)[1]
            for num in pred:
                t = int(num.__getitem__(0))
                test_y.write(str(t) + "\n")


# Plot the graphs
def plotGraphs(model ,modelOptimizer):
    # The y of the real function
    yTrain = []
    # The y of the training set
    yVal = []
    x = []
    # We want our graph to be more accurate so we calcualte 100 points
    for epoch in range(1, 10 + 1):
        print("Epoch number ", epoch)
        train(epoch, model, modelOptimizer, train_loader)
        # Checking the average loss of the train
        avgTrainLoss = avgLossCheck(model, train_loader, "Training")

        avgValidationLoss = avgLossCheck(model, validation_loader, "Validation")
        test(test_loader, model)
        # Adding to x list
        x.append(epoch)
        yTrain.append(avgTrainLoss)
        yVal.append(avgValidationLoss)
    # Making the headers
    plt.title("Average Loss")
    plt.xlabel("Number of epoch")
    plt.ylabel("Average Loss")
    # Plotting the graph
    plt.plot(x,yTrain, color = 'blue')
    plt.plot(x, yVal, color = 'orange')
    # For convenience showing which graph is the real and which is from training
    plt.legend(('Avg Loss Training', 'Avg Loss Validation'), fontsize='small')
    # Showing the graphs
    plt.show()

def main():
    # First model
    modelA = Net()
    modelAOptimizer = optim.SGD(modelA.parameters(), lr=eta)
    plotGraphs(modelA, modelAOptimizer)

    # # # Plotting the graph for model A
    plotGraphs(modelA, modelAOptimizer)

    modelB = DropoutNet(0.1)
    modelBOptimizer = optim.SGD(modelB.parameters(), lr=eta)
    plotGraphs(modelB, modelBOptimizer)

    modelC = BatchNet()
    modelCOptimizer = optim.SGD(modelC.parameters(), lr=eta)
    plotGraphs(modelC, modelCOptimizer)

    # Writing the results on the test.pred file
    writePred(modelC)


if __name__ == '__main__':
    main()


