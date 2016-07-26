# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import os
import random
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
    all_category = list('02345678aAbBcdDeEfFGhHiIjJkLmMnNOpqQrRstTuvwxyYz')
    data = np.empty((), dtype='float32')
    data = np.empty((2892, 1, 28, 28), dtype="float32")
    label = np.empty((2892,), dtype="uint8")
    i = 0
    for root, sub_dirs, files in os.walk(os.path.join(os.getcwd(), 'category')):
        for sub_dir in sub_dirs:
            imgs = os.listdir(os.path.join(root, sub_dir))
            for img in imgs:
                img_path = os.path.join(root, sub_dir, img)
                img_ori = Image.open(img_path)
                arr = np.asarray(img_ori, dtype="float32")
                data[i, :, :, :] = arr
                label[i] = all_category.index(sub_dir)
                i += 1
    return data, label


def train():
    data, label = load_data()

    # 打乱数据
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    print(data.shape[0], ' samples')
    print('all label:', set(label))

    label = np_utils.to_categorical(label, 48)
    model = Sequential()

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 28, 28), border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(16, 3, 3, input_shape=(1, 24, 24), border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, input_shape=(1, 11, 11), border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(input_dim=32*4*4, output_dim=256, init='glorot_uniform'))
    model.add(Activation('tanh'))

    model.add(Dense(input_dim=256, output_dim=128, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))

    model.add(Dense(input_dim=128, output_dim=48, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    check_pointer = ModelCheckpoint('./train_chars.d5', monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(data, label, batch_size=100, nb_epoch=130, verbose=1,
              show_accuracy=True, validation_split=0.2, callbacks=[check_pointer])


if __name__ == '__main__':
    train()
