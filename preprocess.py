#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
#-----------------------------------------------------------------------------------------
#                              对需要处理的验证码的图片进行预处理(作样本)
# 操作:
#   二值化，顶格，字符分割等
# 依赖:
#   openCV(python)[http://opencv.org/], skimage(scikit-image)[http://scikit-image.org/]
#   numpy[http://www.numpy.org/], pandas[http://pandas.pydata.org/]
# 库版本:
#   openCV > 2.4.x, skimage >= 0.9.x
# author:
#   gavin zhou (1964427613@qq.com)
# date:
#   2016/07/17
# version:
#   0.1
#------------------------------------------------------------------------------------------
"""

import cv2
import os
import shutil
import numpy as np
import pandas as pd


def binary_img(img_path):
    """
    二值化
    :param img_path: 原始图像的路径
    :return: 二值化之后的图像
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    retval__, binary_im = cv2.threshold(img, 180, 1, cv2.THRESH_BINARY)
    return binary_im


def ding_ge(binary_im):
    """
    对图像进行顶格
    :param binary_im: 二值化的图像
    :return:
    """
    for x in xrange(0, binary_im.shape[0]):
        line_val = binary_im[x]
        # 不全部是白色(1）
        if not (line_val == 1).all():
            line_start = x
            break
    for x1 in xrange(binary_im.shape[0]-1, -1, -1):
        line_val = binary_im[x1]
        # 不全部是白色(1）
        if not (line_val == 01).all():
            line_end = x1
            break
    for y in xrange(0, binary_im.shape[1]):
        col_val = binary_im[:, y]
        # 不全部是白色(1）
        if not (col_val == 1).all():
            col_start = y
            break
    for y1 in xrange(binary_im.shape[1]-1, -1, -1):
        col_val = binary_im[:, y1]
        # 不全部是白色(1）
        if not (col_val == 1).all():
            col_end = y1
            break
    ding_ge_im = binary_im[line_start:line_end, col_start:col_end]
    # ding_ge_im = binary_im[:, col_start:col_end]
    return ding_ge_im


def vertical_sep(dingge_im, im_name):
    """
    对顶格之后的图像进行垂直切分为若干个小图像
    :param dingge_im: 顶格之后的图像
    :return:
    """
    # 找出所有列不全为白色的index
    not_all_white_index = []
    for x in xrange(0, dingge_im.shape[1]):
        col_val = dingge_im[:, x]
        # 不全部是白色(1）
        if not (col_val == 1).all():
            not_all_white_index.append(x)

    # 找出分段的点
    sep_index = [0]
    for y in xrange(0, len(not_all_white_index)-1):
        if (not_all_white_index[y+1] - not_all_white_index[y]) > 3:  # 剔除可能因为的噪点产生的干扰(间距过小)
            sep_index.append(not_all_white_index[y])
            sep_index.append(not_all_white_index[y+1])
    sep_index.append(not_all_white_index[-1])

    # 进行分割
    save_dir = os.path.join(os.getcwd(), 'sep')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_index = str(im_name.split('.')[0])
    for z in xrange(0, len(sep_index)-1, 2):
        sep_im = dingge_im[:, sep_index[z]:sep_index[z+1]]
        save_name = img_index + '_' + str(z/2+1) + '.png'
        # 如果超过40 pixel
        if (sep_index[z+1] - sep_index[z]) >= 41:
            cv2.imwrite(os.path.join(os.getcwd(), 'sep41', save_name), sep_im*255)
        else:
            cv2.imwrite(os.path.join(save_dir, save_name), sep_im*255)


def region_split(img_path):
    """
    对图片进行连通域的分割处理(CFS)
    :param img_path: 图片路径
    :return:
    """
    from skimage.measure import regionprops
    from skimage.morphology import label
    import math

    im_bw = binary_img(img_path)
    label_image, num_of_region = label(im_bw, neighbors=8, return_num=True)

    coordinates_of_all_region = np.zeros((num_of_region, 4))  # 存储区域的矩形坐标
    # centroid_of_all_region = np.zeros((num_of_region, 2), dtype=float)  # 存储区域的质心坐标
    pixels_of_all_region = np.zeros(num_of_region)  # 存储区域的像素个数

    index = 0
    for region in regionprops(label_image):
        # 跳过包含像素过少的区域
        if region.area < 30:
            continue
        minr, minc, maxr, maxc = region.bbox
        # coord_r, coord_c = region.centroid
        coordinates_of_all_region[index] = np.array([minr, minc, maxr, maxc])
        # centroid_of_all_region[index] = np.array([coord_r, coord_c])
        pixels_of_all_region[index] = region.area

        index += 1

    for x in xrange(0, coordinates_of_all_region.shape[0]):
        # 跳过全是０的(包含像素过少的区域)
        if (coordinates_of_all_region[x] == 0).all():
            continue
        minr, minc, maxr, maxc = coordinates_of_all_region[x]
        # 判断X代表的区域是否被包含
        for y in xrange(0, coordinates_of_all_region.shape[0]):
            if (coordinates_of_all_region[y] == 0).all() or (y == x):
                continue
            minr_, minc_, maxr_, maxc_ = coordinates_of_all_region[y]
            if minr >= minr_ and maxr <= maxr_ and minc >= minc_ and maxc <= maxc_:
                coordinates_of_all_region[x] = np.zeros((1, 4)) - 1  # 被包含的区域矩形坐标置为-1
                # centroid_of_all_region[x] = np.zeros((1, 2)) - 1  # 被包含的区域的质心坐标置为-1
                pixels_of_all_region[x] = -1  # 被包含的区域的像素数目置为-1
                break
    # 删除所有的被包含的区域
    mask = np.array([True] * coordinates_of_all_region.shape[0])
    for i in xrange(0, coordinates_of_all_region.shape[0]):
        if (coordinates_of_all_region[i] == -1).all() or (coordinates_of_all_region[i] == 0).all():
            mask[i] = False
    coordinates_of_all_region = coordinates_of_all_region[mask]
    # centroid_of_all_region = centroid_of_all_region[mask]
    pixels_of_all_region = pixels_of_all_region[mask]

    for x in xrange(0, coordinates_of_all_region.shape[0]):
        if pixels_of_all_region[x] <= 55:  # 像素数目过少，将其合并到最近的区域
            min_dis = 99999
            min_region_index = 9999
            combine_x = 9999
            for y in xrange(0, coordinates_of_all_region.shape[0]):
                if y == x:
                    continue
                if math.fabs(coordinates_of_all_region[x][3] - coordinates_of_all_region[y][3]) < min_dis:
                    min_dis = math.fabs(coordinates_of_all_region[x][3] - coordinates_of_all_region[y][3])
                    min_region_index = y
                    combine_x = x
            # 合并到y代表的区域中
            min_r = min(coordinates_of_all_region[combine_x][0], coordinates_of_all_region[min_region_index][0])
            min_c = min(coordinates_of_all_region[combine_x][1], coordinates_of_all_region[min_region_index][1])
            max_r = max(coordinates_of_all_region[combine_x][2], coordinates_of_all_region[min_region_index][2])
            max_c = max(coordinates_of_all_region[combine_x][3], coordinates_of_all_region[min_region_index][3])
            coordinates_of_all_region[min_region_index] = np.array([min_r, min_c, max_r, max_c])
            coordinates_of_all_region[combine_x] = np.array([-1, -1, -1, -1])
            # centroid_of_all_region[combine_x] = np.array([-1, -1])
            pixels_of_all_region[combine_x] = -1

    # 删除所有的需要合并的区域
    mask1 = np.array([True] * coordinates_of_all_region.shape[0])
    for i in xrange(0, coordinates_of_all_region.shape[0]):
        if (coordinates_of_all_region[i] == -1).all() or (coordinates_of_all_region[i] == 0).all():
            mask1[i] = False
    coordinates_of_all_region = coordinates_of_all_region[mask1]
    # centroid_of_all_region = centroid_of_all_region[mask1]
    pixels_of_all_region = pixels_of_all_region[mask1]

    # 有部分的图像存在定位重合的现象(宽度为高度50)，进行合并
    all_done = False
    for x in xrange(0, coordinates_of_all_region.shape[0]):
        if not all_done:
            for y in xrange(0, coordinates_of_all_region.shape[0]):
                if y == x:
                    continue
                if (coordinates_of_all_region[x][0] == coordinates_of_all_region[y][0] == 0) and (coordinates_of_all_region[x][2] == coordinates_of_all_region[y][2] == 50):
                    # minrow_x = minrow_y and maxrow_x = maxrow_y
                    # 合并两个区域
                    min_r_2 = coordinates_of_all_region[x][0]
                    min_c_2 = min(coordinates_of_all_region[x][1], coordinates_of_all_region[y][1])
                    max_r_2 = coordinates_of_all_region[x][2]
                    max_c_2 = min(coordinates_of_all_region[x][3], coordinates_of_all_region[y][3])
                    coordinates_of_all_region[x] = np.array([-1, -1, -1, -1])
                    coordinates_of_all_region[y] = np.array([min_r_2, min_c_2, max_r_2, max_c_2])
                    all_done = True
                    break
    mask2 = np.array([True] * coordinates_of_all_region.shape[0])
    for i in xrange(0, coordinates_of_all_region.shape[0]):
        if (coordinates_of_all_region[i] == -1).all() or (coordinates_of_all_region[i] == 0).all():
            mask2[i] = False
    coordinates_of_all_region = coordinates_of_all_region[mask2]
    # 对获得的区域坐标按照column的index进行排序
    coordinates_of_all_region = np.array(sorted(coordinates_of_all_region, key=lambda p: p[1]))
    # # 存储图像
    # img_index = im_name.split(".")[0]
    # # print img_index
    # for x in xrange(0, coordinates_of_all_region.shape[0]):
    #     save_img_path = os.path.join(os.getcwd(), 'all_DFS', img_index + "_" + str(x+1) + ".png")
    #     cv2.imwrite(save_img_path, im_bw[coordinates_of_all_region[x][0]:coordinates_of_all_region[x][2],
    #                                coordinates_of_all_region[x][1]:coordinates_of_all_region[x][3]] * 255)
    return coordinates_of_all_region


def make_train_char_db(ori_img_dir):
    import pandas as pd
    import h5py
    """
    均分四份，制作训练集
    :param ori_img_dir: 原始的图像的目录
    :return:
    """
    even_split_train_path = os.path.join(os.getcwd(), 'evensplit_train_im')
    if not os.path.exists(even_split_train_path):
        os.makedirs(even_split_train_path)

    train_imgs = os.listdir(ori_img_dir)
    letters = list('02345678abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    answer_data = pd.read_table(os.path.join(os.getcwd(), 'lvy_ans.txt'), sep=':', names=['Index', 'Answer'])
    # 保存数据
    img = np.zeros((len(train_imgs)*4, 1, 35, 35), dtype=np.uint8)
    label = np.zeros((len(train_imgs)*4), dtype=np.uint32)
    index = 0
    for train_img in train_imgs:
        ori_train_img = os.path.join(ori_img_dir, train_img)
        binary_train_img = binary_img(ori_train_img)  # 二值化之后的图像
        dingge_train_img = ding_ge(binary_train_img)  # 顶格之后的图像

        # 均分成四份
        step_train = dingge_train_img.shape[1] / float(4)
        start_train = [j for j in np.arange(0, dingge_train_img.shape[1], step_train).tolist()]
        for p, k in enumerate(start_train):
            print train_img + '_' + str((p+1))
            split_train_img = dingge_train_img[:, k:k + step_train]
            small_img = ding_ge(split_train_img)
            split_train_resize_img = cv2.resize(small_img, (35, 35))
            img[index, 0, :, :] = split_train_resize_img
            label[index] = letters.index(answer_data['Answer'][int(train_img.split('.')[0])-1][p])
            index += 1
            cv2.imwrite(os.path.join(even_split_train_path,
                                     train_img.split('.')[0] + '_' + str(p+1) + '.png'), split_train_resize_img*255)
    f = h5py.File(os.path.join(os.getcwd(), 'train_chars_data.h5'), 'w')
    f.create_dataset('img', data=img)
    f.create_dataset('label', data=label)
    f.close()


def resize_al_img():
    save_dir = os.path.join(os.getcwd(), 'All_len_resized(3_4)')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    index = 1
    for root, sub_dirs, files in os.walk(os.path.join(os.getcwd(), 'train_len')):
        for sub_dir in sub_dirs:
            imgs = os.listdir(os.path.join(root, sub_dir))
            for img in imgs:
                len_of_img = sub_dir
                img_path = os.path.join(root, sub_dir, img)
                bw = binary_img(img_path)
                resize_bw = cv2.resize(bw, (130, 55))
                save_path = os.path.join(save_dir, str(index) + '_' + len_of_img + '.png')
                cv2.imwrite(save_path, resize_bw * 255)
                index += 1


def equal_split(img_path, num):
    """
    将图像均分为num份
    :param img_path: 图像的路径
    :param num: 份数
    :return:
    """
    save_dir = os.path.join(os.getcwd(), 'all_single_500')
    bw = binary_img(img_path)
    img_name = os.path.basename(img_path)
    step = bw.shape[1] / float(num)
    start = [i for i in np.arange(0, bw.shape[1], step).tolist()]
    for i, c in enumerate(start):
        split_img = bw[:, c:c + step]
        cv2.imwrite(os.path.join(save_dir, img_name.split('.')[0] + '_' + str(i+1) + '.png'), split_img*255)


def classify_to_category():
    """
    按照答案将分割的小图片进行归类
    :return:
    """
    ans = pd.read_table(os.path.join(os.getcwd(), 'lvy_ans.txt'), names=['name', 'ans'], sep=':')
    category_path = os.path.join(os.getcwd(), 'category')
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    all_imgs = os.listdir(os.path.join(os.getcwd(), 'reindex_resized'))
    for img in all_imgs:
        img_path = os.path.join(os.getcwd(), 'reindex_resized', img)
        img_sub_index = int(img.split('.')[0].split('_')[-1]) - 1
        category_of_img = ans.ix[int(img.split('_')[0]) - 1]['ans'][img_sub_index]
        save_category_path = os.path.join(category_path, category_of_img)
        if not os.path.exists(save_category_path):
            os.makedirs(save_category_path)
        save_img_path = os.path.join(save_category_path, img)
        shutil.copyfile(img_path, save_img_path)

if __name__ == '__main__':
    # CFS
    # ori_img_path = os.path.join(os.getcwd(), 'ori_im_to_train')
    # # 存储处理后的图像
    # all_imgs = os.listdir(ori_img_path)
    # for img_name in all_imgs:
    #     ori_img = os.path.join(ori_img_path, img_name)
    #     region_split(ori_img, img_name)  # CFS

    # split all image
    # ans = pd.read_table('./num.txt', names=['name', 'ans'], sep=':')
    # imgs = os.listdir(os.path.join(os.getcwd(), 'all_DFS'))
    # for img in imgs:
    #     length_of_img = 1
    #     img_path = os.path.join(os.getcwd(), 'all_DFS', img)
    #     for x in xrange(0, len(ans['ans'])):
    #         if ans.ix[x]['name'] == img:
    #             length_of_img = ans.ix[x]['ans']
    #             break
    #     if length_of_img != 1:
    #         equal_split(img_path, length_of_img)
    #     else:
    #         shutil.copyfile(img_path, os.path.join(os.getcwd(), 'all_single_500', img))

    # category
    classify_to_category()

