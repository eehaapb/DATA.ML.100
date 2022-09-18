import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import randint, random

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

def load_cifar10(num):
    x = []
    y = []

    for i in range(1, num):
        X, Y = load_cifar10_batch('cifar-10-batches-py/data_batch_%d' % (i,))
        x.append(X)
        y.append(Y)

    Xtr = np.concatenate(x)
    Ytr = np.concatenate(y)
    del X, Y
    Xte, Yte = load_cifar10_batch('cifar-10-batches-py/test_batch')
    
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
    TP = ((pred == 1) & (gt == 1)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    precision = TP / (TP+FP)

    return accuracy

def cifar10_classifier_random(x):
    rc = []

    for i in range(len(x)-1):
        rc.insert(i, randint(0,9))

    return rc

def cifar10_classifier_1nn(tstdata,trdata,trlabels):
    
    n = trdata.shape[0]
    # n = 100
    pred_1nnc = np.zeros(n, dtype=trlabels.dtype)

    for i in range(n):
        distance = np.sqrt(np.sum(np.square(trdata - tstdata[i,:]), axis=1))
        min_index = np.argmin(distance)
        # print(min_index)
        pred_1nnc[i] = trlabels[min_index]

    return pred_1nnc


def show_images(X, Y, label_names):
    for i in range(X.shape[0]):
    # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)


def main():

    Xtr, Ytr, Xte, Yte = load_cifar10(2)
    

    labeldict = unpickle('cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    # Yte_pred = cifar10_classifier_1nn(Xte_rows,Xtr_rows,Ytr)
    Yte_rnd = cifar10_classifier_random(Yte)
    Yte_1nn_trdata = cifar10_classifier_1nn(Xtr_rows,Xtr_rows,Ytr)
    Yte_1nn = cifar10_classifier_1nn(Xte_rows, Xtr_rows, Ytr)
    print("Random classifier accuracy: ", class_acc(Yte_rnd,Ytr))
    print("Training data used as test data for 1nn classifier accuracy: ",class_acc(Yte_1nn_trdata,Ytr)," (should be 1.0)")
    print("Test batch 1nn accuracy: ", class_acc(Yte_1nn,Ytr))






if __name__ == "__main__":
    main()