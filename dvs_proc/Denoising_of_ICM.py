# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: .\Denoising_of_ICM.py
# Compiled at: 2020-06-09 17:42:45
# Size of source mod 2**32: 1787 bytes
import scipy.io as sio, numpy as np
from PIL import Image
from src import Show_accumulate_pic
from src import FilterTD_Fast
from src import Event_denoise
import h5py

def Denoising(event, image_width, image_height, event_delta, nei_weight, savePath, s):
    print('The original event stream size: ', event.shape)
    accumulate_index, number_images = Show_accumulate_pic.get_accumulate_ts(event, image_width, image_height, event_delta)
    if s == 1:
        print('Visualizing the original event stream........')
        #Show_accumulate_pic.show_pic_denoise(event, image_width, image_height, accumulate_index, number_images, flag='show')
    event= Event_denoise.event_denoise_t_3d(event, image_width, image_height, accumulate_index, number_images,
      nei_weight, flag='ICM')
    sio.savemat(savePath, {'denoisedEvents': event})
    print('The denoised event stream size: ', event.shape)
    
    if s == 1:
        print('Visualizing the denoised event stream........')
        Show_accumulate_pic.show_pic_denoise(event, image_width, image_height, accumulate_index, number_images, flag='noshow')
    print('The denoising is complete and the file is saved.')
