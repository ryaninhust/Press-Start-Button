#!/usr/bin/env python
#encoding: utf-8

import numpy as np
from PIL import Image


def create_data(file):
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip('\n').split(' ')
            feature_dict = {}
            for l in line[1:]:
                ls = l.split(':')
                feature_dict[ls[0]] = ls[1]
            data_list.append([line[0], feature_dict])
    return data_list

def find_m_index(n):
    """find matrix index giving feature index"""
    return (n - 1) / 105, (n - 1) % 105

def find_f_index(x, col):
    """find feature index giving matrix index"""
    return x[0] * col + x[1] + 1

def cut_blank(image, filename):
    feature = image[1]
    # find matrix index and remove noise
    matrix_index = {find_m_index(int(f)):float(feature[f]) for f in feature
                    if float(feature[f]) > 0.35}
    if matrix_index:
        row_index = [m[0] for m in matrix_index]
        col_index = [m[1] for m in matrix_index]
        matrix_cut = {(m[0] - min(row_index),m[1] - min(col_index)):matrix_index[m]
                       for m in matrix_index}
        col_range = max(col_index) - min(col_index) + 1
        row_range = max(row_index) - min(row_index) + 1
        create_image(filename, matrix_cut, row_range, col_range)
    else:
        create_image(filename, matrix_index, 60, 60)

def create_image(filename, matrix_index, nrow, ncol, normalize = False, t = 0):
    matrix_init = np.zeros((nrow, ncol))
    for i in matrix_index:
        if normalize:
            if float(matrix_index[i]) > t:
                matrix_init[i[0]][i[1]] = 255
        else:
            matrix_init[i[0]][i[1]] = float(matrix_index[i]) * 255
    im = Image.fromarray(matrix_init)
    image_name = 'image/' + filename + '.jpg'
    im.convert('RGB').save(image_name)

def image_preprocessing(image_data, dir_name):
    #image_valid = [image for image in image_data if len(image_data[1])>0]
    for idx, image in enumerate(image_data):
        filename = dir_name + str(idx) + '_' + str(image[0])
        print filename
        cut_blank(image, filename)

if __name__ == "__main__":
    image_train = create_data('ml14fall_train.dat')
    image_test = create_data('ml14fall_test1_no_answer.dat')
    image_preprocessing(image_train, 'train/')
    image_preprocessing(image_test, 'test/')



