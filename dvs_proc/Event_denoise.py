# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\Event_denoise.py
# Compiled at: 2020-06-08 13:21:36
# Size of source mod 2**32: 16477 bytes
import scipy.io as sio, numpy as np
from PIL import Image
from src import Show_accumulate_pic
from src import FilterTD_Fast
from src import ICM_order_denoise
from src import SA_denoise

def event_denoise(event, image_width, image_height, accumulate_index, number_images, flag='ICM'):
    """
    功能：将事件流累积成图像，利用ICM算法对单张图像进行去噪
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return:
    denoise_event：去除噪声之后的事件流
    noise_index：存储噪声事件的行地址索引值
    """
    total_event = event.shape[0]
    binary_image = np.ones((image_width, image_height)) * -1
    noise_index = []
    image_index = 1
    event_index = 0
    while image_index <= number_images:
        print('image_index = ', image_index)
        temp_event_index = event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            binary_image[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
            event_index += 1

        if flag == 'ICM':
            noise_event = ICM_order_denoise.ICM_event(binary_image)
        else:
            if flag == 'SA':
                noise_event = SA_denoise.simulated_annealing_event(binary_image)
            else:
                print('############### please input flag = ICM or flag = SA ###################')
        event_index = temp_event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            current_event = (image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))
            if current_event in noise_event:
                event[(event_index, 3)] = 0
                noise_index.append(event_index)
            event_index += 1

        binary_image[:, :] = -1
        image_index += 1

    denoise_event = FilterTD_Fast.RemoveNulls(event)
    return (
     denoise_event, noise_index)


def event_denoise_t_3d(event, image_width, image_height, accumulate_index, number_images, nei_weight, flag='ICM'):
    """
    功能：利用ICM算法，进行三维事件流去噪
    原理：根据累积节点的时间，累积成帧图像，然后利用3维ICM算法去噪
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return:
    denoise_event：去除噪声之后的事件流（矩阵）
    #noise_index：存储噪声事件在原始数据中的索引值
    """
    total_event = event.shape[0]
    binary_image_1 = np.ones((image_width, image_height)) * -1
    binary_image_2 = np.ones((image_width, image_height)) * -1
    binary_image_3 = np.ones((image_width, image_height)) * -1
    image_index = 1
    event_index = 0
    while image_index <= number_images:
        print('image_index:', image_index)
        temp_event_index = event_index
        if image_index == 1:
            binary_image_1[:] = 0
        else:
            while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1 - 1)]):
                binary_image_1[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
                event_index += 1

        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            binary_image_2[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
            event_index += 1

        if image_index == number_images:
            binary_image_3[:] = 0
        else:
            while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index + 1 - 1)]):
                binary_image_3[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
                event_index += 1

        if flag == 'ICM':
            noise_event = ICM_order_denoise.ICM_event_3d(binary_image_1, binary_image_2, binary_image_3, nei_weight)
        if flag == 'SA':
            noise_event = ICM_order_denoise.SA_event_3d(binary_image_1, binary_image_2, binary_image_3)
        event_index = temp_event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            current_event = (image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))
            if current_event in noise_event:
                event[(event_index, 3)] = 0
            event_index += 1

        binary_image_1[:, :] = -1
        binary_image_2[:, :] = -1
        binary_image_3[:, :] = -1
        image_index += 1

    denoise_event = FilterTD_Fast.RemoveNulls(event)
    return (denoise_event)


def event_denoise_t_3d_fast(event, image_width, image_height, accumulate_index, number_images, flag='ICM'):
    """
    功能：本来想通过保留图片2和3作为下一次循环的图片1和2来提高速度，结果。。。。并没有提高
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return: 返回去除噪声点之后的event,存储噪声事件索引值的noise_index
    """
    total_event = event.shape[0]
    binary_image_1 = np.ones((image_width, image_height)) * -1
    binary_image_2 = np.ones((image_width, image_height)) * -1
    binary_image_3 = np.ones((image_width, image_height)) * -1
    noise_index = []
    image_index = 1
    event_index = 0
    while image_index <= number_images:
        temp_event_index = event_index
        if image_index == 1:
            binary_image_1[:] = 0
            while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
                binary_image_2[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
                event_index += 1

        if image_index == number_images:
            binary_image_3[:] = 0
        else:
            while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index + 1 - 1)]):
                binary_image_3[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
                event_index += 1

        if flag == 'ICM':
            noise_event = ICM_order_denoise.ICM_event_3d(binary_image_1, binary_image_2, binary_image_3)
        event_index = temp_event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            current_event = (image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))
            if current_event in noise_event:
                event[(event_index, 3)] = 0
                noise_index.append(event[(event_index, 5)])
            event_index += 1

        binary_image_1 = binary_image_2
        binary_image_2 = binary_image_3
        binary_image_3[:] = -1
        image_index += 1

    denoise_event = FilterTD_Fast.RemoveNulls(event)
    return (
     denoise_event, noise_index)


def event_denoise_t(event, image_width, image_height, accumulate_index, number_images, flag='ICM'):
    """
    根据记录的累积成帧图片的时间戳，进行累积，然后利用ICM进行事件流去噪
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return: 返回去除噪声点之后的event,存储噪声事件索引值的noise_index
    """
    total_event = event.shape[0]
    binary_image = np.ones((image_width, image_height)) * -1
    noise_index = []
    image_index = 1
    event_index = 0
    while image_index <= number_images:
        temp_event_index = event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            binary_image[(image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))] = 1
            event_index += 1

        if flag == 'ICM':
            noise_event = ICM_order_denoise.ICM_event(binary_image)
        else:
            if flag == 'SA':
                noise_event = SA_denoise.simulated_annealing_event(binary_image)
            else:
                print('############### please input flag = ICM or flag = SA ###################')
        event_index = temp_event_index
        while (event_index < total_event - 1) & (event[(event_index, 3)] <= accumulate_index[(image_index - 1)]):
            current_event = (image_width - 1 - int(event[(event_index, 0)]), int(event[(event_index, 1)]))
            if current_event in noise_event:
                event[(event_index, 3)] = 0
                noise_index.append(event_index)
            event_index += 1

        binary_image[:, :] = -1
        image_index += 1

    denoise_event = FilterTD_Fast.RemoveNulls(event)
    return (
     denoise_event, noise_index)


def event_denoise_3d_amount(event, image_width, image_height, event_delta):
    """
    功能：按照事件数量，设置缓冲区，直接得到三帧图像，然后利用3维ICM算法去噪
    ps：本来想着提高速度，测试之后发现并没有提高。。。。。
    :param event:  输入事件流
    :param image_width: 宽度
    :param image_height: 高度
    :param accumulate_index: 记录的累积成帧图片的时间戳
    :param number_images: 累积成的图片总数量
    :return: 返回去除噪声点之后的event,存储噪声事件索引值的noise_index
    """
    total_event = event.shape[0]
    noise_index = []
    image_index = 1
    number_images = int(total_event / event_delta)
    event_x_buffer = np.zeros((1, event_delta))
    event_y_buffer = np.zeros((1, event_delta))
    binary_image_1 = np.ones((image_width, image_height)) * -1
    binary_image_2 = np.ones((image_width, image_height)) * -1
    binary_image_3 = np.ones((image_width, image_height)) * -1
    while image_index <= number_images:
        if image_index == 1:
            binary_image_1[:] = 0
            start_event_index_2 = (image_index - 1) * event_delta
            stop_event_index_2 = image_index * event_delta
            event_x_buffer[0, :] = event[start_event_index_2:stop_event_index_2, 0]
            event_y_buffer[0, :] = event[start_event_index_2:stop_event_index_2, 1]
            event_index = 0
            while event_index < event_delta:
                binary_image_2[(int(event_y_buffer[(0, event_index)]), int(event_x_buffer[(0, event_index)]))] = 1
                event_index += 1

        if image_index == number_images:
            binary_image_3[:] = 0
        else:
            start_event_index_3 = (image_index + 1 - 1) * event_delta
            stop_event_index_3 = (image_index + 1) * event_delta
            event_x_buffer[0, :] = event[start_event_index_3:stop_event_index_3, 0]
            event_y_buffer[0, :] = event[start_event_index_3:stop_event_index_3, 1]
            event_index = 0
            while event_index < event_delta:
                binary_image_3[(int(event_y_buffer[(0, event_index)]), int(event_x_buffer[(0, event_index)]))] = 1
                event_index += 1

        noise_event = ICM_order_denoise.ICM_event_3d(binary_image_1, binary_image_2, binary_image_3)
        event_index = 0
        while event_index < event_delta:
            current_index = start_event_index_3 - event_delta + event_index
            current_event = (
             int(event[(current_index, 1)]), int(event[(current_index, 0)]))
            if current_event in noise_event:
                event[(event_index, 3)] = 0
                noise_index.append(event[(event_index, 5)])
            event_index += 1

        binary_image_1[:] = binary_image_2[:]
        binary_image_2[:] = binary_image_3[:]
        binary_image_3[:] = -1
        image_index += 1

    denoise_event = FilterTD_Fast.RemoveNulls(event)
    return (
     denoise_event, noise_index)