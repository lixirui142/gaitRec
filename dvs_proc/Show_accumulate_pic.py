# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\Show_accumulate_pic.py
# Compiled at: 2020-06-09 17:39:23
# Size of source mod 2**32: 6887 bytes
import h5py
import scipy.io as sio, numpy as np
from PIL import Image
import cv2
import os

def get_accumulate_ts(event, image_width, image_height, event_delta):
    """
    功能：按照事件数量将事件流累积成帧图像，记录累积的时间节点和累积的图像数量
    :param event: 输入的事件流，矩阵，依次为 x,y,adc,t
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :param event_delta: 累积的事件数
    :return:
    accumulate_index：存储的是累积成帧图片的事件节点，是个列表，存储的是一个个事件戳ts
    number_images: 表示累积成图片的总数
    """
    total_event = event.shape[0]
    binary_image = np.zeros((image_width, image_height))
    image_index = 1
    number_images = int(np.ceil(total_event / event_delta))
    event_x_buffer = np.zeros((1, event_delta))
    event_y_buffer = np.zeros((1, event_delta))
    event_adc_buffer = np.zeros((1, event_delta))
    event_t_buffer = np.zeros((1, event_delta))
    accumulate_index = []
    while image_index <= number_images:
        start_event_index = (image_index - 1) * event_delta
        stop_event_index = image_index * event_delta
        event_x_buffer[0, :] = event[start_event_index:stop_event_index, 1]
        event_y_buffer[0, :] = event[start_event_index:stop_event_index, 0]
        event_adc_buffer[0, :] = event[start_event_index:stop_event_index, 2]
        event_t_buffer[0, :] = event[start_event_index:stop_event_index, 3]
        event_index = 0
        while event_index < event_delta:
            binary_image[(image_width - 1 - int(event_y_buffer[(0, event_index)]), int(event_x_buffer[(0, event_index)]))] = 255
            event_index += 1

        accumulate_index.append(event_t_buffer[(0, event_index - 1)])
        binary_image[:, :] = 0
        image_index += 1

    return (accumulate_index, number_images)


def show_pic_denoise(event, image_width, image_height, accumulate_index, number_images, flag='show'):
    """
    根据记录的累积节点的时间，可视化事件流
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return: 无返回值
    """
    total_event = event.shape[0]
    binary_image = np.zeros((image_width, image_height))
    image_index = 1
    event_index = 0
    while image_index <= number_images:
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            binary_image[(int(image_width - 1 - event[(event_index, 0)]), int(event[(event_index, 1)]))] = 255
            event_index += 1

        image = Image.fromarray(binary_image).convert('1', dither=(Image.NONE))
        cv2.imshow('image', binary_image)
        cv2.waitKey(80)
        if flag == 'show':
            pass
        else:
            path_dir = r'dvs/filter'
            path = os.path.join(path_dir,str(image_index) + '.png')
            image.save(path)
            f = h5py.File('gait0.h5','w')
            f['DVS'] = binary_image
            f.close()


        binary_image[:, :] = 0
        image_index += 1

    cv2.destroyAllWindows()


def show_pic(event, image_width, image_height, event_delta=int(50000), flag='show'):
    """
    根据事件数量累积成帧图像，可视化事件流
    :param event: [m, 4]矩阵，依次表示 y, x, p, t
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :param event_delta: ；累积的脉冲数
    :return: 无返回值
    """
    total_event = event.shape[0]
    binary_image = np.zeros((image_height, image_width))
    image_index = 1
    number_images = total_event / event_delta
    event_x_buffer = np.zeros((1, event_delta))
    event_y_buffer = np.zeros((1, event_delta))
    event_adc_buffer = np.zeros((1, event_delta))
    event_t_buffer = np.zeros((1, event_delta))
    print(max(event[:,0]), max(event[:, 1]))
    while image_index <= number_images:
        start_event_index = (image_index - 1) * event_delta
        stop_event_index = image_index * event_delta
        event_x_buffer[0, :] = event[start_event_index:stop_event_index, 1]
        event_y_buffer[0, :] = event[start_event_index:stop_event_index, 0]
        event_adc_buffer[0, :] = event[start_event_index:stop_event_index, 2]
        event_t_buffer[0, :] = event[start_event_index:stop_event_index, 3]
        event_index = 0
        while event_index < event_delta:
            binary_image[(image_height - 1 - int(event_y_buffer[(0, event_index)]), int(event_x_buffer[(0, event_index)]))] = 255
            event_index += 1

        image = Image.fromarray(binary_image).convert('1', dither=(Image.NONE))
        cv2.imshow('image', binary_image)
        cv2.waitKey(80)
        if flag == 'show':
            pass
        else:
            path_dir = r'dvs'
            path = os.path.join(path_dir,str(image_index) + '.png')
            image.save(path)
        binary_image[:, :] = 0
        image_index += 1

    cv2.destroyAllWindows()


def main():
    data = sio.loadmat('../data/TennisPlayground.mat')
    event = data['eventMatrix']
    image_width = 641
    image_height = 769
    event_delta = int(50000)
    show_pic(event, image_width, image_height, event_delta)
    print('#############')


if __name__ == '__main__':
    main()