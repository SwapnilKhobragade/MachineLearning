# Eliad Arzuan 206482622
from __future__ import print_function
import torch
from torchvision import transforms, datasets, models
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torchvision import utils
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
import time
import copy


val_percent = 0.2
batch_size = 50
eta = 0.001
criterion = nn.CrossEntropyLoss()

# Getting the device: cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is ", device)

# Transform the image into normalized tensor
normalizedTransofmrs = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Transform the image into normalized tensor
resizedTransforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def loadDataSet(transforms=normalizedTransofmrs):
    cifar10 = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transforms)

    num_train = len(cifar10)
    indices = list(range(num_train))
    split = int(val_percent * num_train)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)

    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=cifar10,
                                               batch_size=batch_size, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset=cifar10,
                                                    batch_size=batch_size, sampler=validation_sampler)

    # Loading the train set
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False)

    return train_loader, validation_loader, test_loader


def loadDataResnet():
    train_loader, validation_loader, test_loader = loadDataSet()


classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Loading our data
train_loader, validation_loader, test_loader = loadDataSet()

# Loading data for resnet:
resnet_train, resnet_validation, resnet_test = loadDataSet(transforms=resizedTransforms)

first_hidden = 100
second_hidden = 50


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, first_hidden)
        # Adding first batch_norm
        self.batch_norm1 = nn.BatchNorm1d(first_hidden)

        self.fc2 = nn.Linear(first_hidden, second_hidden)
        # Adding second batch_norm
        self.batch_norm2 = nn.BatchNorm1d(second_hidden)
        # 10 classes
        self.fc3 = nn.Linear(second_hidden, 10)

    def forward(self, x):
        # i. Conv Layer
        x = self.conv1(x)
        # ii. ReLU activation function
        x = F.relu(x)
        # iii. Pooling layer
        x = self.pool(x)
        # iv. Conv Layer
        x = self.conv2(x)
        # v. ReLU activation function
        x = F.relu(x)
        # vi. Pooling layer
        x = self.pool(x)
        # vii. Fully connected + batch norm
        x = x.view(-1, 16 * 5 * 5)
        x = self.batch_norm1(self.fc1(x))
        # viii. ReLU activation function + dropout
        x = F.relu(x)
        # ix. Fully connected + batch norm
        x = self.batch_norm2(self.fc2(x))
        # x. ReLU activation function
        x = F.relu(x)
        # xi. Fully connected
        x = self.fc3(x)
        # We will use another loss cross entropy that in pytorch implementation includes softmax
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Training the model
def train(net, optimizer):
    net.train()

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Finished Training')


def avgLossCheck(model, loader, name, confusion=False):
    model.eval()
    loss = 0.0
    correct = 0.0
    yPred = []
    yTest = []
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss += float(criterion(output, target).item())  # sum up batch loss
        # get the index of the max log-probability - argmax
        pred = output.data.max(1, keepdim=True)[1]

        # If we don't want confusion matrix we don't have to waste so many time
        if (confusion):
            for numPred, numTest in zip(pred, target):
                t1 = int(numPred.__getitem__(0))
                t2 = int(numTest)
                yPred.append(classes[t1])
                yTest.append(classes[t2])
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    loss /= float(len(loader) * batch_size)
    print('\n', name, ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader) * batch_size,
                       100. * correct / (len(loader) * batch_size)))

    if (confusion):
        print("\n")
        print("Confusion matrix")
        matrix = confusion_matrix(y_true=yTest, y_pred=yPred, labels=classes)
        # Printing the matrix with headers
        print(pd.DataFrame(matrix, columns=classes, index=classes))
        #  print(pd.crosstab(yTest, yPred, rownames=classes, colnames=classes, margins=True))

    return loss


epochs = 15


# Testing the model
def test(test_loader, model, confusion=False):
    avgLossCheck(model, test_loader, "Test", confusion=confusion)


def writePred(model):
    model.eval()
    # Write into file
    with open('test.pred', 'w') as test_y:
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            # get the index of the max log-probability - argmax
            pred = output.data.max(1, keepdim=True)[1]
            for num in pred:
                t = int(num.__getitem__(0))
                test_y.write(str(t) + "\n")


# Plot the graphs
def plotGraphs(model, modelOptimizer, trainLoader=train_loader, validationLoader=validation_loader
               , testLoader=test_loader, num_ephocs = 1):
    # The y of the real function
    yTrain = []
    # The y of the training set
    yVal = []
    x = []
    # We want our graph to be more accurate so we calcualte 100 points
    for epoch in range(1, num_ephocs + 1):
        print("Epoch number ", epoch)
        train(model, modelOptimizer)
        # Checking the average loss of the train
        avgTrainLoss = avgLossCheck(model, trainLoader, "Training")

        avgValidationLoss = avgLossCheck(model, validationLoader, "Validation")
        if (epoch == num_ephocs):
            test(testLoader, model, confusion=True)
        else:
            test(testLoader, model)
        # Adding to x list
        x.append(epoch)
        yTrain.append(avgTrainLoss)
        yVal.append(avgValidationLoss)
    # Making the headers
    plt.title("Average Loss")
    plt.xlabel("Number of epoch")
    plt.ylabel("Average Loss")
    # Plotting the graph
    plt.plot(x, yTrain, color='blue')
    plt.plot(x, yVal, color='orange')
    # For convenience showing which graph is the real and which is from training
    plt.legend(('Avg Loss Training', 'Avg Loss Validation'), fontsize='small')
    # Showing the graphs
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs=1):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dataloaders = {'train': resnet_train, 'val':resnet_validation}
        dataset_sizes = {'train': len(resnet_train) * batch_size, 'val':len(resnet_validation) * batch_size}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def plotResnet(model, trainLoader = resnet_train, validationLoader = resnet_validation
               , testLoader=test_loader, num_ephocs = 1):
    # The y of the real function
    yTrain = []
    # The y of the training set
    yVal = []
    x = []
    # We want our graph to be more accurate so we calcualte 100 points
    for epoch in range(1, num_ephocs + 1):
        # Checking the average loss of the train
        avgTrainLoss = avgLossCheck(model, trainLoader, "Training")

        avgValidationLoss = avgLossCheck(model, validationLoader, "Validation")
        if (epoch == num_ephocs):
            test(testLoader, model, confusion=True)
        else:
            test(testLoader, model)
        # Adding to x list
        x.append(epoch)
        yTrain.append(avgTrainLoss)
        yVal.append(avgValidationLoss)
    # Making the headers
    plt.title("Average Loss")
    plt.xlabel("Number of epoch")
    plt.ylabel("Average Loss")
    # Plotting the graph
    plt.plot(x, yTrain, color='blue')
    plt.plot(x, yVal, color='orange')
    # For convenience showing which graph is the real and which is from training
    plt.legend(('Avg Loss Training', 'Avg Loss Validation'), fontsize='small')
    # Showing the graphs
    plt.show()
def main():
    # cNet = Net()
    # cNet = cNet.to(device)
    # optimizer = optim.Adam(cNet.parameters(), lr=eta)
    # plotGraphs(cNet, optimizer, num_ephocs = epochs)
    #
    # writePred(cNet)

    # We need to freeze all the network except the final layer. We need to set requires_grad == False
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 10)


    model_conv = model_conv.to(device)


    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=1)

    print("\nNow checking stats")
    plotResnet(model=model_conv)

if __name__ == '__main__':
    main()
