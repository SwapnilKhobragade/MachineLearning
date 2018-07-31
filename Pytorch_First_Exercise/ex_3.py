import numpy as np
import pickle
import os.path
from random import shuffle

#####################
# Eliad Arzuan
# 206482622
#####################


# Hyper parameters
ephochs = 20
eta = 0.01
# Size of the hidden layer
hidden = 100

#Like in the rec we will do sigmoid in lambda expression
sigmoid = lambda x: 1/(1+np.exp(-x))
# The derivative of the sigmoid
dSigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))


# The input is wx+b
softmax = lambda x: np.exp(x - np.max(x)) / (np.exp(x - np.max(x))).sum()

def Loss(yhat, y):
    # yhat contains a vector of probabilities so we need to choose the probability of the current y
    prob = yhat[int(y)]
    return -np.log(prob)



def initializeWeights():
    # Initialize random parameters and inputs
    # We will initialize by uniform distribution - works the best
    # W1 - with size of hidden x 784
    W1 = np.random.uniform(-0.08,0.08,[hidden,784])
    # b1- with size of hidden x 1
    b1 = np.random.uniform(-0.08,0.08,[hidden, 1])
    # W2 with size of 10 x hidden (we have 10 classes)
    W2 = np.random.uniform(-0.08,0.08, [10, hidden])
    # b2 with size of 10 x 1
    b2 = np.random.uniform(-0.08,0.08, [10, 1])
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# Shuffles two arrays together
def shuffleTogether(a, b):
    c = list(zip(a, b))
    shuffle(c)
    a, b = zip(*c)
    return a,b

def loadTrainFiles(x ="train_x", y = "train_y"):
    print "Loading the training files"
    train_x = np.loadtxt(x)
    train_y = np.loadtxt(y)
    # Shuffles our training files
    train_x, train_y = shuffleTogether(train_x, train_y)
    return train_x, train_y

def loadTestFile(x = "test_x"):
    print "Loading the test file.."
    return np.loadtxt(x)

# Forward propogation
def fprop(x, y, params):
  # Getting all the parameters
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  # Now we want that the dimentions of x will fit with the w
  # W1 is hidden x 784 so we want x  will be 784 x 1 instead of just a vector

  x.shape = (784, 1)
  z1 = np.dot(W1, x) + b1
  # We will move it to the next layer with sigmoid
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  # We will do softmax of w2x+b2
  h2 = softmax(z2)
  # Loss like we got in the tirgul
#  loss = -(y * np.log(h2) + (1-y) * np.log(1-h2))
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}#, 'loss': loss}
  # We want to save the results
  for key in params:
    ret[key] = params[key]
  return ret

# Backward propagation - update the parameters
def bprop(fprop_cache):
  # Get the values from the last running and update by them
  x, y, z1, h1, z2, h2 = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
  # Create vector of zeroes
  yZ = np.zeros(10)
  yZ.shape = (10, 1)
  yZ[int(y)] = 1

# Derivatives by the chain rule
  dz2 = (h2 - yZ)  # dL/dh2
  dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
  db2 = dz2  # dL/dz2 * dz2/db2

  dh1 = np.dot(fprop_cache['W2'].T, dz2)
  dz1 = dh1 * dSigmoid(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1

  dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1

  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def updateWeights(params, bprop_cache):
    W1, b1, W2, b2 = [params[key] - eta * bprop_cache[key]  for key in ('W1', 'b1', 'W2', 'b2')]

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }


def predictProbs(val_x, val_y, params):
    correct = 0.0
    for x,y in zip(val_x, val_y):
        calc = fprop(x, y, params)
        yhat = calc['h2']
        yhat.shape = (10)

        # Counting correct guesses
        if np.argmax(yhat) == y:
            correct += 1
    # Get the percentage of accuracy
    return (correct / val_x.shape[0]) * 100

# Training the network
def train(params, train_x, train_y, val_x, val_y):
    print "Start with training:"
#    loss = 0.0
    for i in range(0, ephochs):
        print "Ephoc number " + str(i)
        # Shuffeling to prevent overfeeting
        train_x, train_y = shuffleTogether(train_x, train_y)

        for x, y in zip(train_x, train_y):
            # Calculate with current weights
            fprop_cache = fprop(x, y, params)
            # Getting the gradients
            bprop_cache = bprop(fprop_cache)

            # Update the gradients
            params = updateWeights(params, bprop_cache)
    return params



def main():
    # Loading our files
    train_x, train_y = loadTrainFiles()

    test_x = loadTestFile()

    #Normalize the test values
    test_x = np.divide(test_x, 255.0)

    # We want to normalize the pixels before using sigmoid
    train_x = np.divide(train_x, 255.0)
    # Getting the training set size
    train_size = len(train_y)

    print "size of original file is " + str(train_size)

    print "Size of test file is " + str(len(test_x))
    # Our validation set
    val_size = int(train_size * 0.2)
    # Taking 20% of the training to the validation set
    val_x, val_y = train_x[-val_size:], train_y[-val_size:]


    # We will use only 80% of the data in the validation set
    train_x, train_y = train_x[val_size:], train_y[val_size:]



    # print val_x.shape
    # print train_x.shape


    print "Initialize parameters"
    # Getting the parameters
    params = initializeWeights()

    params = train(params, train_x, train_y, val_x, val_y)
    print "Finished train"

    # Checking the validation set
    print "Now checking your accurancy for the validation set"
    acc = predictProbs(val_x, val_y, params)
    print "You were right on " + str(acc) + "%"

    print "Now Writing the result for test_x into file"
    print "Scanning test_x"

    # Write into file
    with open('test.pred', 'w') as test_y:
        for x in test_x:
            # We dont need a y because we dont do backprop
            calc = fprop(x, 1, params)
            y = np.argmax(calc['h2'])
            test_y.write(str(y) + "\n")

    print "Finished!"



if __name__=='__main__':
    main()

