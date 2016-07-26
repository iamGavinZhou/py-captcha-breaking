# -*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
#                        破解验证码的演示程序
# 依赖:
#   Keras[http://keras.io/]
# 库版本:
#   Keras >= 0.2.x
# author:
#   gavin zhou (1964427613@qq.com)
# date:
#   2016/7/25
# version:
#   0.1
#-----------------------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import cv2
import time
import sys
import shutil
import preprocess
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def get_size_phase(width):
    import math
    """
    根据图片长度判断可能的字符长度
    :param width: 图片的长度
    :return:
    """

    '''
    典型的字符长度为:
    单个字符 23pixel，两个字符59pixel，三个字符105pixel，四个字符140pixel，宽度超过145pixel认为是4个字符
    此判断方法存在较大的误差，但暂时我还没想到很好的方法，sorry!
    有更好的方法的欢迎联系我，一起交流学习哈，邮箱1964427613@qq.com
    '''
    if width >= 145:
        return 4
    # 计算和典型长度的差值
    dis_23 = math.fabs(width - 23)
    dis_59 = math.fabs(width - 59)
    dis_105 = math.fabs(width - 105)
    dis_140 = math.fabs(width - 140)
    # 相邻两个距离差之和最小的
    phase1_dis_sum = dis_23 + dis_59
    phase2_dis_sum = dis_59 + dis_105
    phase3_dis_sum = dis_105 + dis_140
    min_dis_sum = min([phase1_dis_sum, phase2_dis_sum, phase3_dis_sum])
    if phase1_dis_sum == min_dis_sum:
        return 1  # 可能的字符个数是1或者2
    elif phase2_dis_sum == min_dis_sum:
        return 2  # 可能的字符个数是2或者3
    else:
        return 3  # 可能的字符个数是3或者4


def initial_num_char_phase1():
    """
    识别二值化图像的字符个数
    :param bw: 二值图像
    :return:
    """
    # 加载模型
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

    # 加载权值
    model.load_weights('model/train_len_size1.d5')

    sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    return model


def initial_num_char_phase2():
    """
    识别二值化图像的字符个数
    :param bw: 二值图像
    :return:
    """
    # 加载模型
    model = Sequential()

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 40, 100), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 36, 96), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 5, 5, input_shape=(1, 32, 92), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(Dropout(0.45))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 14, 29), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 11, 26), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(Convolution2D(16, 4, 4, input_shape=(1, 8, 23), border_mode='valid'))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    model.add(Flatten())
    model.add(Dense(input_dim=16*2*10, output_dim=256, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.65))

    model.add(Dense(input_dim=256, output_dim=128, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.40))

    model.add(Dense(input_dim=128, output_dim=2, init='glorot_uniform'))
    model.add(Activation('softmax'))

    # 加载权值
    model.load_weights('model/train_len_size2.d5')

    sgd = SGD(l2=0.0, lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    return model


def initial_num_char_phase3():
    """
    识别二值化图像的字符个数
    :param bw: 二值图像
    :return:
    """
    # 加载模型
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

    # 加载权值
    model.load_weights('model/train_len_size3.d5')

    sgd = SGD(l2=0.0, lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    return model


def initial_classify_single_char():
    """
    识别二值化图像的字符(单个)
    :param bw: 二值图像
    :return:
    """
    # 加载模型
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

    # 加载权值
    model.load_weights('model/train_chars.d5')

    sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    return model


def main():
    # 初始化相关网络
    print('Start demo....')
    print('Networks initialization....')
    print('Initial network1....')
    phase1_model = initial_num_char_phase1()
    print('Network1 initialization done....')
    print('Initial network2....')
    phase2_model = initial_num_char_phase2()
    print('Network2 initialization done....')
    print('Initial network3....')
    phase3_model = initial_num_char_phase3()
    print('Network3 initialization done....')
    # chars_model = initial_classify_single_char()

    print('start timing....')
    start_time = time.clock()
    test_dir = os.path.join(os.getcwd(), 'demo')  # 测试图像的目录
    # demo_res = os.path.join(os.getcwd(), 'demo_result.txt')  # 答案文件
    # fid = open(demo_res, 'w')
    all_test_imgs = os.listdir(test_dir)
    sorted_all_test_imgs = np.array(sorted(all_test_imgs, key=lambda x: int(x.split('.')[0])))
    temp_demo_path = os.path.join(os.getcwd(), 'demo_all_img')
    if not os.path.exists(temp_demo_path):
        os.makedirs(temp_demo_path)
    for test_img in sorted_all_test_imgs:
        # fid.write(test_img + ":")
        test_img_path = os.path.join(test_dir, test_img)
        index_temp = 1
        # 预处理阶段
        # step 1 CFS
        coordinates_of_test_img = preprocess.region_split(test_img_path)  # CFS图像块的坐标
        # step 2 获得各个CFS块的字符个数
        bw = preprocess.binary_img(test_img_path)
        for index in xrange(0, coordinates_of_test_img.shape[0]):
            minr, minc, maxr, maxc = coordinates_of_test_img[index]  # 图像块矩形区域的坐标(左上角和右下角)
            cfs_block_img = bw[minr:maxr, minc:maxc]  # CFS图像块
            # 保存临时结果
            # save_temp_path = os.path.join(os.getcwd(), 'temp', test_img.split('.')[0] + '_' + str(index) + '.png')
            # cv2.imwrite(save_temp_path, cfs_block_img*255)
            phase_of_test_img = get_size_phase(maxc - minc)
            if phase_of_test_img == 1:
                # 1或者2个字符
                cfs_block_img = cv2.resize(cfs_block_img, (40, 30))
                num_of_chars = phase1_model.predict_classes(
                    cfs_block_img.astype(np.float32).reshape((1, 1, 30, 40)) / 255, verbose=0) + 1
            elif phase_of_test_img == 2:
                # 2或者3个字符
                cfs_block_img = cv2.resize(cfs_block_img, (100, 40))
                num_of_chars = phase2_model.predict_classes(
                    cfs_block_img.astype(np.float32).reshape((1, 1, 40, 100)) / 255, verbose=0) + 2
            elif phase_of_test_img == 3:
                # 3或者4个字符
                cfs_block_img = cv2.resize(cfs_block_img, (130, 55))
                num_of_chars = phase3_model.predict_classes(
                    cfs_block_img.astype(np.float32).reshape((1, 1, 55, 130)) / 255, verbose=0) + 3
            elif phase_of_test_img == 4:
                # 4个字符
                num_of_chars = 4
            # step 3 均分
            step = cfs_block_img.shape[1] / float(num_of_chars)
            start = [i for i in np.arange(0, cfs_block_img.shape[1], step).tolist()]
            for i, c in enumerate(start):
                split_img = cfs_block_img[:, c:c + step]
                # step 4 保存单个字符
                split_img = cv2.resize(split_img, (28, 28))
                save_demo_img_path = os.path.join(temp_demo_path,
                                                  test_img.split('.')[0] + '_' + str(index_temp) + '.png')
                cv2.imwrite(save_demo_img_path, split_img*255)
                index_temp += 1
                '''
                可以不借助caffe只使用keras来识别单个字符
                不知道为何，keras识别有问题，任何单个字符都识别为i且validation accuracy只有0.85左右
                故此保存了单个字符，使用caffe下的mnist网络识别
                '''
                # category_index = chars_model.predict_classes(
                #     split_img.astype(np.float32).reshape((1, 1, 28, 28)) / 255, verbose=0)  # 单个字符的类index
                # char_res = all_category[category_index]
                # fid.write(char_res)
    #     fid.write('\n')
    # fid.close()
    end_time = time.clock()
    print('Split for {0} images, has used {1}s'.format(len(all_test_imgs), round((end_time - start_time), 3)))


def caffe_clasify_single_char():
    """
    使用caffe下的mnist网络识别单个字符,accuracy 89%左右
    :return:
    """
    print('\nStart caffe image classification....')
    time_start = time.clock()

    # caffe路径
    caffe_root = '/home/gavinzhou/caffe-master/'
    sys.path.insert(0, caffe_root + 'python')

    test_dir = os.path.join(os.getcwd(), 'demo_all_img')
    answer = os.path.join(os.getcwd(), 'lvy_demo.txt')

    netProPath = os.path.join(os.getcwd(), 'data', 'lenet.prototxt')
    modelPath = os.path.join(os.getcwd(), 'model', 'lvy_iter_9000.caffemodel')
    sysTXTPath = os.path.join(os.getcwd(), 'data', 'class_index.txt')

    import caffe
    caffe.set_mode_cpu()  # GPU模式
    net = caffe.Net(netProPath, modelPath, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1,0))

    net.blobs['data'].reshape(1, 3, 28, 28)

    output = open(answer, 'w')

    # 读取标签
    labels = np.loadtxt(sysTXTPath, str, delimiter=':')

    # 识别
    for root, dirs, files in os.walk(test_dir):
        for i in xrange(0, files.__len__()):
            sf = os.path.join(root, files[i])
            output.write(((files[i]).split('.'))[0] + ':'),
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(sf))
            out = net.forward()
            index = out['prob'].argmax()  # 类别的index
            output.write((labels[index][1]) + '\n')
    output.close()

    time_end = time.clock()
    print('All single image classification done....')
    print('Has runed {0}s'.format(round((time_end - time_start), 3)))


def sort_res():
    """
    转换生成的答案文件
    :return:
    """
    import pandas as pd
    result_dir = os.path.join(os.getcwd(), 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_txt = os.path.join(result_dir, 'demo_recog.txt')
    fid = open(save_txt, 'w')
    res_path = os.path.join(os.getcwd(), 'lvy_demo.txt')
    res = pd.read_table(res_path, names=['index', 'char'], sep=':')
    array_of_res = res['index'].values
    list_of_res = array_of_res.tolist()
    for x in xrange(1, 51):
        fid.write(str(x) + ":")
        for y in xrange(1, 5):
            # 找到index
            to_find = str(x) + '_' + str(y)
            if to_find in list_of_res:
                index_of_char = list_of_res.index(to_find)
                char_res = res.ix[index_of_char]['char']
                fid.write(char_res)
        fid.write('\n')
    fid.close()


def clean():
    """
    清理工作
    :return:
    """
    print('\nClean start up....')
    shutil.rmtree(os.path.join(os.getcwd(), 'demo_all_img'))  # 删除保存单个字符的目录
    os.remove(os.path.join(os.getcwd(), 'lvy_demo.txt'))  # 删除临时文件
    print('Clean done...')


if __name__ == '__main__':
    main()
    caffe_clasify_single_char()
    sort_res()
    clean()
    print('Demo all done....')

