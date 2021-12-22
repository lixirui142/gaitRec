# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\SA_denoise.py
# Compiled at: 2020-06-08 13:24:44
# Size of source mod 2**32: 7631 bytes
from random import random
import time, os, numpy as np
from PIL import Image
beta = 0.001
eta = 0.0021
h = 0.0
kmax = 15

def global_E(x, y):
    r"""
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


def temperature(k, kmax):
    """Schedule the temperature for simulated annealing."""
    return 0.002 * (1.0 / k - 1.0 / kmax)


def prob(E_old, E_new, t):
    """Probability transition function for simulated annealing."""
    if E_old > E_new:
        return 1
    else:
        return np.exp((E_old - E_new) / t)


def localized_E(E_old, i, j, x, y):
    """
    针对某个像素值[i, j],根据旧状态下的能量E_old，然后将状态取反，计算新状态下的能量E_new
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


def simulated_annealing(y, kmax=int(15)):
    """
    利用模拟退火算法进行去噪
    :param y: 带噪图像数字矩阵，尺寸[m, n]，范围[-1 1]
    :param kmax: 迭代次数
    :return: x: 去噪之后的数字图像矩阵，尺寸[m, n]，范围[-1 1]
    """
    x = np.array(y)
    Ebest = Ecur = global_E(x, y)
    for k in range(1, kmax + 1):
        t = temperature(k, kmax + 1)
        for idx in np.ndindex(y.shape):
            old_value, new_value, E_old, E_new = localized_E(Ecur, idx[0], idx[1], x, y)
            p, q = prob(E_old, E_new, t), random()
            if p > q:
                Ecur, x[idx] = E_new, new_value
                if E_new < Ebest:
                    Ebest = E_new
            else:
                Ecur, x[idx] = E_old, old_value

    return x


def simulated_annealing_event(y, kmax=int(15)):
    """
    利用累积的一定数量的事件个数，模拟退火算法进行事件流去噪
    :param y: 输入的[-1 1]矩阵
    :param kmax: 迭代次数
    :return: noise_event: 存储的为判定为噪声事件的位置信息
    """
    x = np.array(y)
    Ebest = Ecur = global_E(x, y)
    noise_event = []
    for k in range(1, kmax + 1):
        t = temperature(k, kmax + 1)
        for idx in np.ndindex(y.shape):
            old_value, new_value, E_old, E_new = localized_E(Ecur, idx[0], idx[1], x, y)
            p, q = prob(E_old, E_new, t), random()
            if p > q:
                Ecur, x[idx] = E_new, new_value
                if E_new < Ebest:
                    Ebest = E_new
            else:
                Ecur, x[idx] = E_old, old_value

        print('k=', k)

    for idx in np.ndindex(y.shape):
        if y[idx] == 1 and x[idx] == -1:
            noise_event.append(idx)

    return noise_event


def sign(data, translate):
    """Map a dictionary for the element of data.

    Example:
        To convert every element in data with value 0 to -1, 255 to 1,
        use `signed = sign(data, {0: -1, 255: 1})`
    """
    temp = np.array(data)
    return np.vectorize(lambda x: translate[x])(temp)


def main():
    image = Image.open('../img/single_ball.jpg')
    image_binary = image.convert('1', dither=(Image.NONE))
    data = sign(image_binary.getdata(), {0:-1,  255:1})
    y = data.reshape(image.size[::-1])
    result = simulated_annealing(y, kmax)
    result = sign(result, {-1: 0, 1: 255})
    output_image = Image.fromarray(result).convert('1', dither=(Image.NONE))
    output_image.show()
    output_image.save('../img/single_ball_denoise_1.png', 'PNG')
    print('############ DENOISING VICTORY ###########')


if __name__ == '__main__':
    main()