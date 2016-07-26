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
    data = np.empty((100, 1, 55, 130), dtype="float32")
    label = np.empty((100,), dtype="uint8")
    imgs = os.listdir("./train_len/All_len_resized(3_4)")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./train_len/All_len_resized(3_4)/" + imgs[i])
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        label[i] = int((imgs[i].split('.')[0]).split('_')[-1]) - 3
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

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 55, 130), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 51, 126), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 47, 122), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.45))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 21, 39), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 18, 36), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 15, 33), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Flatten())
    model.add(Dense(input_dim=16*6*15, output_dim=256, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.60))

    model.add(Dense(input_dim=256, output_dim=128, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.40))

    model.add(Dense(input_dim=128, output_dim=2, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(l2=0.0, lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    check_pointer = ModelCheckpoint('./train_len_size3.d5', monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(data, label, batch_size=100, nb_epoch=100, verbose=1,
              show_accuracy=True, validation_split=0.2, callbacks=[check_pointer])


if __name__ == '__main__':
    train()
