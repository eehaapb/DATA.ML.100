import numpy as np
import pickle

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding="latin1")

        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
        Y = np.array(Y)

        return X, Y

def load_cifar10(num):
    x = []
    y = []

    for i in range(1, num):
        X, Y = load_cifar10_batch('cifar-10-batches-py/data_batch_%d' % (i,))
        x.append(X)
        y.append(Y)

    Xtr = np.concatenate(x)
    Ytr = np.concatenate(y)

    Xte, Yte = load_cifar10_batch('cifar-10-batches-py/test_batch')
    
    return Xtr, Ytr, Xte, Yte

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

Xtr, Ytr, Xte, Yte = load_cifar10(2) # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

# Now that we have all images stretched out as rows, here is how we could train and evaluate a classifier:

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
acc = np.mean(Yte_predict == Yte)

print("Accuracy: ", acc)
