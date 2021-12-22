# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\Show_time_pic.py
# Compiled at: 2020-06-09 17:39:23
# Size of source mod 2**32: 1885 bytes
import scipy.io as sio, numpy as np
from PIL import Image
import cv2

def show_pic(event, image_width, image_height, FPS=24, timeconst=1e-06):
    """
    累积固定时间间隔的事件数，进行可视化操作
    :param event: [m, 4]矩阵，依次表示，y, x, p, t
    :param timeconst: 微妙
    :param FPS: 帧率
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :return:
    """
    total_event = event.shape[0]
    FrameLength = 1 / (FPS * timeconst)
    time_interval = event[(0, 3)] + FrameLength
    binary_image = np.zeros((image_width, image_height))
    event_index = 0
    while event_index < total_event - 1:
        while event[(event_index, 3)] <= time_interval:
            binary_image[(int(event[(event_index, 1)]), int(event[(event_index, 0)]))] = 255
            if event_index < total_event - 1:
                event_index += 1
            else:
                break

        image = Image.fromarray(binary_image)
        cv2.imshow('image', binary_image)
        cv2.waitKey(80)
        binary_image[:, :] = 0
        time_interval += FrameLength

    cv2.destroyAllWindows()


def main():
    data = sio.loadmat('../data/single_man.mat')
    event = data['eventMatrix']
    timeconst = 1e-06
    FPS = 2400
    image_width = 641
    image_height = 769
    show_pic(event, image_width, image_height, FPS, timeconst)


if __name__ == '__main__':
    main()