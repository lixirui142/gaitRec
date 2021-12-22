# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\FilterTD_Fast.py
# Compiled at: 2020-06-08 13:24:44
# Size of source mod 2**32: 5784 bytes
import scipy.io as sio, numpy as np
from PIL import Image
from src import Show_time_pic
from src import Show_accumulate_pic

def RemoveNulls(event):
    """
    根据设定的单个噪声事件的ts设置为0，进行删除该噪声事件所在的行
    :param event: 事件矩阵[m, 4]
    :return: 删除噪声事件之后的事件流
    """
    denoise_event = event[(event[:, 3] != 0)]
    return denoise_event


def FilterTD_Fast(event, image_width, image_height, us_Time):
    """
    功能：从原始事件流中删除噪声事件，同时记录噪声事件在原始数据中的行坐标索引
    原理：设定单个事件点周围3*3像素值，设定固定事件窗大小，如果在该时间窗内，周围邻近像素存在事件点，
    那么该事件为有效事件，否则为噪声事件，设置其t==0，然后调用RemoveNulls删除该事件[矩阵删除该行]
    ps：由于要记录噪声事件的在原始数据中的行地址，所以添加列index，记录下原始的事件行地址
    :param event: 事件流，矩阵,x、y、adc、t，添加了列index
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :param us_Time: 时间窗
    :return:
    event：去除掉噪声事件的事件流，矩阵
    noise_index：噪声事件在原始事件流（矩阵）中的行坐标索引
    """
    event[:, 3] += us_Time
    total_event = event.shape[0]
    T0_max = np.zeros((image_width + 2, image_height + 2))
    noise_index = []
    index = 0
    while index < total_event:
        event_x = event[(index, 0)]
        event_y = event[(index, 1)]
        event_t = event[(index, 3)]
        pre_val = T0_max[(event_y, event_x)]
        adjacent_list = [
         -1, 0, 1]
        for x_d in adjacent_list:
            for y_d in adjacent_list:
                T0_max[(event_y + y_d, event_x + x_d)] = max(T0_max[(event_y + y_d, event_x + x_d)], event_t)

        T0_max[(event_y, event_x)] = pre_val
        if event_t >= T0_max[(event_y, event_x)] + us_Time:
            event[(index, 3)] = 0
            noise_index.append(event[(index, 5)])
        index += 1

    event = RemoveNulls(event)
    event[:, 3] = event[:, 3] - us_Time
    return (
     event, noise_index)


def FilterTD_Fast_original(event, image_width, image_height, us_Time):
    event[:, 3] += us_Time
    total_event = event.shape[0]
    T0_max = np.zeros((image_width + 2, image_height + 2))
    index = 0
    while index < total_event:
        event_x = event[(index, 0)]
        event_y = event[(index, 1)]
        event_t = event[(index, 3)]
        pre_val = T0_max[(event_y, event_x)]
        adjacent_list = [
         -1, 0, 1]
        for x_d in adjacent_list:
            for y_d in adjacent_list:
                T0_max[(event_y + y_d, event_x + x_d)] = max(T0_max[(event_y + y_d, event_x + x_d)], event_t)

        T0_max[(event_y, event_x)] = pre_val
        if event_t >= T0_max[(event_y, event_x)] + us_Time:
            event[(index, 3)] = 0
        index += 1

    event = RemoveNulls(event)
    event[:, 3] = event[:, 3] - us_Time
    return event


def main():
    data = sio.loadmat('../data/TennisPlayground.mat')
    event = data['eventMatrix']
    image_width = 641
    image_height = 769
    us_Time = 500
    accumulate_index, number_images = Show_accumulate_pic.get_accumulate_ts(event, image_width, image_height)
    denoise_event = FilterTD_Fast_original(event, image_width, image_height, us_Time)
    Show_accumulate_pic.show_pic_denoise(denoise_event, image_width, image_height, accumulate_index, number_images)
    print('############ denoist victory #############')


if __name__ == '__main__':
    main()