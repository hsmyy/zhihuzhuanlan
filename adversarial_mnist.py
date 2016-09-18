from __future__ import absolute_import
#from __future__ import print_function
import os
import struct
from array import array

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np
import matplotlib.pyplot as plt

class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        self.test_images = np.array(ims)
        self.test_labels = np.array(labels)

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                         os.path.join(self.path, self.train_lbl_fname))

        self.train_images = np.array(ims)
        self.train_labels = np.array(labels)

        np.random.seed(1337)
        np.random.shuffle(self.train_images)
        np.random.seed(1337)
        np.random.shuffle(self.train_labels)

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0]*rows*cols)

        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print ("Showing num:" , train_label[0])
        print (self.display(train_img[0]))
        print
        return True

    @classmethod
    def display(cls, img, width=28):
        render = ''
        for i in range(len(img)):
            if i % width == 0: render += '\n'
            if img[i] > 200:
                render += '1'
            else:
                render += '0'
        return render

def image_generator(img, batch_size):
    dataset = np.zeros((64, 1, 28, 28))
    for i in range(batch_size):
        dataset[i] = img + np.random.uniform(low=-0.1, high=0.1, size=(1, 28, 28))
    return dataset

def build_model():
    nb_classes = 10
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1,28,28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='normal')) 
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def gen(X_train, Y_train, X_test, Y_test):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20

    img = X_train[2]

    img = img.astype("float32").reshape((1,28,28))
    label = Y_train[2]
    img /= 255.0
    print 'label=' + str(label)
    plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray)
    plt.show()

    model = build_model()
    model.load_weights('mnist_cnn')

    for iterator in range(200):
        
        ds = image_generator(img, 64)    
        pred = model.predict(ds, batch_size=64)
        pred_label = np.argmax(pred, axis=1)
        flag = False
        for i in range(64):
            if pred_label[i] == label:
                choosed_img = ds[i]
                flag = True
                break
        if flag == False:
            print 'iter=' + str(iterator) + ", break"
            break
        else:
            img = choosed_img
            print 'iter=' + str(iterator) + ", label = " + str(label)
            if iterator == 50 or iterator == 100 or iterator == 150:
                plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray)
                plt.show()
    print img
    plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray)
    plt.show()

        # choose the best
def gen2(X_train, Y_train, X_test, Y_test):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20

    img = X_train[2]

    img = img.astype("float32").reshape((1,28,28))
    label = Y_train[2]
    img /= 255.0
    print 'label=' + str(label)

    model = build_model()
    model.load_weights('mnist_cnn')

    for iterator in range(1000):
        ds = image_generator(img, 64)    
        pred = model.predict(ds, batch_size=64)
        pred_label = np.argmax(pred, axis=1)
        flag = False
        for i in range(64):
            if pred_label[i] != label:
                choosed_idx = i
                flag = True
                break
        if flag == False:
            print 'iter=' + str(iterator) + ", no change"
            img = ds[0]
        else:
            img = ds[choosed_idx]
            print 'iter=' + str(iterator) + ", label = " + str(pred_label[choosed_idx])
            break
    plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray)
    plt.show()


def CNN(X_train, Y_train, X_test, Y_test):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20
    
    X_train = X_train.reshape(60000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1, 28, 28)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    print(X_train.shape, 'train samples')
    print(Y_train.shape, 'train labels')
    print(X_test.shape, 'test smaples')

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = build_model()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=30)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    model.save_weights('mnist_cnn')
    print('Test score:', score)

if __name__ == "__main__":
    print ('Testing')
    mn = MNIST('.')
    if mn.test():
        print ('Passed')
        #CNN(mn.train_images, mn.train_labels, mn.test_images, mn.test_labels)
        gen2(mn.train_images, mn.train_labels, mn.test_images, mn.test_labels)
