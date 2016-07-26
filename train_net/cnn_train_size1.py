# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import os
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range


def load_data():
    data = np.empty((), dtype='float32')
    data = np.empty((5268, 1, 30, 40), dtype="float32")
    label = np.empty((5268,), dtype="uint8")
    imgs = os.listdir("./train_len/All_len_resized(1_2)")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./train_len/All_len_resized(1_2)/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        label[i] = int((imgs[i].split('.')[0]).split('_')[-1]) - 1
    return data, label


def train():
    data, label = load_data()
    # 打乱数据
    index = [i for i in range(len(data))]
    data = data[index]
    label = label[index]
    print(data.shape[0], ' samples')
    print(set(label))

    label = np_utils.to_categorical(label, 2)

    model = Sequential()
    model.add(Convolution2D(4, 5, 5, input_shape=(1, 30, 40), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 26, 36), border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 11, 16), border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.60))

    model.add(Flatten())
    model.add(Dense(input_dim=16*4*6, output_dim=256, init='glorot_uniform'))
    model.add(Activation('tanh'))

    model.add(Dense(input_dim=256, output_dim=2, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    check_pointer = ModelCheckpoint('./train_len_size1.d5', monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(data, label, batch_size=100, nb_epoch=50, verbose=1,
              show_accuracy=True, validation_split=0.2, callbacks=[check_pointer])


if __name__ == '__main__':
    train()
