# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\ICM_order_denoise.py
# Compiled at: 2020-06-08 13:24:44
# Size of source mod 2**32: 12804 bytes
from random import random
import time, os, numpy as np
from PIL import Image
beta = 0.001
eta = 0.0021
h = 0.0
kmax = 2

def temperature(k, kmax):
    """Schedule the temperature for simulated annealing."""
    return 0.002 * (1.0 / k - 1.0 / kmax)


def prob(E_old, E_new, t):
    """Probability transition function for simulated annealing."""
    if E_old > E_new:
        return 1
    else:
        return np.exp((E_old - E_new) / t)


def global_E(x, y):
    """
    计算矩阵x, y的全局能量

    Usage: global_E = global_E(x, y)
    Formula:
        E = h * \sum{x_i} - beta * \sum{x_i x_j} - eta * \sum{x_i y_i}

    :param x: 目前图像矩阵，尺寸：[m ,n]，像素值：[-1, 1]
    :param y: 带噪图像矩阵, 尺寸：[m, n]，像素值：[-1, 1]
    :return: global_E: 全局能量
    """
    x_neighbor = np.zeros_like(x)
    x_neighbor[:-1, :] = x[1:, :]
    x_neighbor[1:, :] += x[:-1, :]
    x_neighbor[:, :-1] += x[:, 1:]
    x_neighbor[:, 1:] += x[:, :-1]
    x_product_x = np.sum(x_neighbor * x)
    x_product_y = np.sum(x * y)
    x_sum = np.sum(x)
    global_E = h * x_sum - beta * x_product_x - eta * x_product_y
    return global_E


def is_valid(i, j, shape):
    """
    判断像素值[i, j]是否超出矩阵范围
    :param i: 像素值坐标
    :param j: 像素值坐标
    :param shape: 图像矩阵尺寸[m, n]
    :return: valid_flag: True or False
    """
    valid_flag = False
    if i >= 0:
        if j >= 0:
            if i < shape[0]:
                if j < shape[1]:
                    valid_flag = True
    return valid_flag


def localized_E(E_old, i, j, x, y):
    """
    功能：计算帧图像的二维局部能量
    原理：针对某个像素值[i, j],根据旧状态下的能量E_old，然后将状态取反，计算新状态下的能量E_new
    ps：注意，这里是在全局能量的基础上进行局部更新的，并没有重新计算整个图像矩阵的全局能量

    Usage：old_value, new_value, E_old, E_new = localized_E(E_old, i, j, x, y)

    :param E_old: 旧状态能量
    :param i: 单个像素坐标
    :param j: 单个像素坐标
    :param x: 目标图像矩阵，尺寸[m, n]，像素值[-1, 1]
    :param y: 带噪图像矩阵，尺寸[m, n],像素值[-1, 1]
    :return: old_value: 旧状态像素值
             new_value: 新状态像素值
             E_old:    旧状态的能量
            E_new:    新状态的能量
    """
    old_value = x[(i, j)]
    new_value = old_value * -1
    adjacent_pixel = [
     (0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = [x[(i + di, j + dj)] for di, dj in adjacent_pixel if is_valid(i + di, j + dj, x.shape)]
    E_new = E_old - h * old_value + h * new_value
    E_new = E_new + eta * y[(i, j)] * old_value - eta * y[(i, j)] * new_value
    E_new = E_new + beta * sum(a * old_value for a in neighbors)
    E_new = E_new - beta * sum(a * new_value for a in neighbors)
    return (
     old_value, new_value, E_old, E_new)


def ICM_event(y):
    """
    利用ICM进行去噪，记录判定为噪声事件的位置信息
    :param y: 输入的[-1, 1]矩阵
    :return: noise_event：存储的为判定为噪声事件的位置信息
    """
    x = np.array(y)
    Ebest = Ecur = global_E(x, y)
    noise_event = []
    for idx in np.ndindex(y.shape):
        old_value, new_value, E_old, E_new = localized_E(Ecur, idx[0], idx[1], x, y)
        if E_new < Ebest:
            Ecur, x[idx] = E_new, new_value
            Ebest = E_new
            if old_value == 1:
                noise_event.append(idx)
        else:
            Ecur, x[idx] = E_old, old_value

    return noise_event


def localized_E_3d(E_old, i, j, x_pre, x_current, x_post, y, nei_weight):
    """
    功能：计算中间图像单个事件的三维局部能量
    :param E_old: 就状态能量
    :param i: 单个像素横坐标
    :param j: 单个像素纵坐标
    :param x_pre: 第一帧图像
    :param x_current: 第二帧图像（去噪的图像）
    :param x_post: 第三帧图像
    :param y: 第二帧图像对应的带噪图像
    :return:
    old_value：旧状态值
    new_value：新状态值
    E_old：旧状态能量
    E_new：新状态能量
    """
    old_value = x_current[(i, j)]
    new_value = old_value * -1
    adjacent_pixel_pre = [
     (0, 0)]
    adjacent_pixel_current = [
     (0, 1), (0, -1), (1, 0), (-1, 0)]
    adjacent_pixel_post = [
     (0, 0)]
    neighbors_pre = [x_pre[(i + di, j + dj)] * nei_weight for di, dj in adjacent_pixel_pre if is_valid(i + di, j + dj, x_pre.shape)]
    neighbors_current = [x_current[(i + di, j + dj)] for di, dj in adjacent_pixel_current if is_valid(i + di, j + dj, x_current.shape)]
    neighbors_post = [x_post[(i + di, j + dj)] * nei_weight for di, dj in adjacent_pixel_post if is_valid(i + di, j + dj, x_post.shape)]
    neighbors = neighbors_pre + neighbors_current + neighbors_post
    E_new = E_old - h * old_value + h * new_value
    E_new = E_new + eta * y[(i, j)] * old_value - eta * y[(i, j)] * new_value
    E_new = E_new + beta * sum(a * old_value for a in neighbors)
    E_new = E_new - beta * sum(a * new_value for a in neighbors)
    return (
     old_value, new_value, E_old, E_new)


def ICM_event_3d(x_pre, x_current, x_post, nei_weight):
    """
    功能：通过比较旧状态与新状态的能量，判定噪声事件
    :param x_pre: 第一帧图像
    :param x_current: 第二帧图像
    :param x_post: 第三帧图像
    :return:
    noise_event：存储噪声事件在当前帧图像（第二帧）的横纵坐标
    """
    y = np.array(x_current)
    Ebest = Ecur = 100.0
    noise_event = []
    for idx in np.ndindex(y.shape):
        old_value, new_value, E_old, E_new = localized_E_3d(Ecur, idx[0], idx[1], x_pre, x_current, x_post, y, nei_weight)
        if E_new < Ebest:
            Ecur, x_current[idx] = E_new, new_value
            Ebest = E_new
            if old_value == 1:
                noise_event.append(idx)
        else:
            Ecur, x_current[idx] = E_old, old_value

    return noise_event


def SA_event_3d(x_pre, x_current, x_post):
    """
    功能：通过比较旧状态与新状态的能量，判定噪声事件
    :param x_pre: 第一帧图像
    :param x_current: 第二帧图像
    :param x_post: 第三帧图像
    :return:
    noise_event：存储噪声事件在当前帧图像（第二帧）的横纵坐标
    """
    y = np.array(x_current)
    Ebest = Ecur = 100.0
    noise_event = []
    for k in range(1, kmax + 1):
        t = temperature(k, kmax + 1)
        for idx in np.ndindex(y.shape):
            old_value, new_value, E_old, E_new = localized_E_3d(Ecur, idx[0], idx[1], x_pre, x_current, x_post, y)
            p, q = prob(E_old, E_new, t), random()
            if p > q:
                Ecur, x_current[idx] = E_new, new_value
                if E_new < Ebest:
                    Ebest = E_new
            elif old_value == 1:
                noise_event.append(idx)
            else:
                Ecur, x_current[idx] = E_old, old_value

    return noise_event


def ICM(y):
    """
    功能：利用ICM对单张图片去噪
    :param y: 二维像素矩阵[-1 ,1]
    :return: x: 去噪之后的二维像素矩阵[-1, 1]
    """
    x = np.array(y)
    Ebest = Ecur = global_E(x, y)
    for idx in np.ndindex(y.shape):
        old_value, new_value, E_old, E_new = localized_E(Ecur, idx[0], idx[1], x, y)
        if E_new < Ebest:
            Ecur, x[idx] = E_new, new_value
            Ebest = E_new
        else:
            Ecur, x[idx] = E_old, old_value

    return x


def sign(data, translate):
    """Map a dictionary for the element of data.

    Example:
        To convert every element in data with value 0 to -1, 255 to 1,
        use `signed = sign(data, {0: -1, 255: 1})`
    """
    temp = np.array(data)
    return np.vectorize(lambda x: translate[x])(temp)


def main():
    image = Image.open('../img/paper.jpg')
    image_binary = image.convert('1', dither=(Image.NONE))
    data = sign(image_binary.getdata(), {0:-1,  255:1})
    y = data.reshape(image.size[::-1])
    result = ICM(y)
    result = sign(result, {-1: 0, 1: 255})
    output_image = Image.fromarray(result).convert('1', dither=(Image.NONE))
    output_image.show()
    output_image.save('../img/paper_denoising.png', 'PNG')
    print('############ DENOISING VICTORY ###########')


if __name__ == '__main__':
    main()