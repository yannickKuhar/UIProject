import sys
import time
import numpy as np

from sklearn import svm
from skimage import color
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from keras.datasets import cifar100, fashion_mnist, mnist

TAG = '[MAIN]'
TAG_ERROR = '[ERROR]'

SVM = 'SVM'
RF = 'RF'
NB = 'NB'
MC = 'MC'

MNIST = 'MNIST'
FMNIST = 'FMNIST'
CIFAR100 = 'CIFAR100'


def majority_classifier(classes, test):
    count_dict = {k: list(classes).count(k) for k in set(classes)}
    return [sorted(count_dict, key=count_dict.get, reverse=True)[0]] * len(test)


def load_data(Data):
    print(f'{TAG} Data load start.')
    if Data == FMNIST:
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        X = np.concatenate((train_images, test_images))
        Y = np.concatenate((train_labels, test_labels))
        X = np.array([f.flatten() for f in X]) / 255
    elif Data == MNIST:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        X = np.concatenate((train_images, test_images))
        Y = np.concatenate((train_labels, test_labels))
        X = np.array([f.flatten() for f in X]) / 255
    elif Data == CIFAR100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_train = np.array([color.rgb2gray(x).flatten() for x in x_train])
        x_test = np.array([color.rgb2gray(x).flatten() for x in x_test])
        X = np.concatenate((x_train, x_test))
        Y = np.concatenate((y_train, y_test))
        Y = Y.flatten()
    else:
        print(TAG, TAG_ERROR, 'Dataset not supported.')

    print(f'{TAG} Data load done.')

    return X, Y,


def split_data(x, y):
    print(f'{TAG} Data split start.')
    ss = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(x, y))

    test_index = ss[0][0]
    train_index = ss[0][1]

    x_train, x_test = np.array(x)[train_index], np.array(x)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    print(f'{TAG} Data split end.')

    return x_train, x_test, y_train, y_test


def main(argv):
    start = time.time()

    if argv[1] == SVM:
        clf = svm.SVC(gamma=0.001, verbose=True)
    elif argv[1] == RF:
        clf = RandomForestClassifier(max_depth=15, random_state=0, verbose=True)
    elif argv[1] == NB:
        clf = GaussianNB()
    elif argv[1] == MC:
        pass
    else:
        print(f'{TAG} {TAG_ERROR} Invalid ML model.')

    X, Y = load_data(argv[2])

    x_train, x_test, y_train, y_test = split_data(X, Y)

    print(f'{TAG} Model: {argv[1]} training start.')

    if argv[1] == MC:
        predictions = majority_classifier(y_train, y_test)
    else:
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)

    print(f'{TAG} CA: {round(accuracy_score(predictions, y_test), 2)}')
    print(f'{TAG} Time: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main(sys.argv)