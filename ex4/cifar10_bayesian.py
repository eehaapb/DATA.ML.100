import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from random import random


def unpickle(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding="latin1")

        return datadict

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding="latin1")

        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
        Y = np.array(Y)

        return X, Y

def load_cifar10():
    x = []
    y = []

    for i in range(1, 6):
        X, Y = load_cifar10_batch('../cifar-10-batches-py/data_batch_%d' % (i,))
        x.append(X)
        y.append(Y)

    Xtr = np.concatenate(x)
    Ytr = np.concatenate(y)
    del X, Y
    Xte, Yte = load_cifar10_batch('../cifar-10-batches-py/test_batch')
    
    return Xtr, Ytr, Xte, Yte

def class_acc2(pred,gt):
    pred_labels = []  # Will store the N predicted class-IDs
    for row in pred:
        pred_labels.append(np.argmax(row))

    correct = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == gt[i]:
            correct += 1


    return correct / len(pred)

def class_acc(pred, gt):
    N = gt.shape[1]
    accuracy = (gt == pred).sum() / N

    return accuracy

def show_images(X, Y, label_names):
    for i in range(X.shape[0]):
    # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)

def cifar10_color(X):
    
    Xrs = np.zeros((X.shape[0],3))
    for i in range(len(X) - 1):
        Xrs[i] = transform.resize(X[i], (1,1))

    return Xrs

def main():

    Xtr, Ytr, Xte, Yte = load_cifar10()
    Xtr_rs = cifar10_color(Xtr)
    Xte_rs = cifar10_color(Xte)

    print(Xte_rs.shape)
    print(Xtr_rs.shape)
    
    labeldict = unpickle('../cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    # show_images(Xtr_rs,Ytr,label_names)

    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


if __name__ == "__main__":
    main()