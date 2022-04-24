from src import read
import os
from aermanager.aerparser import load_events_from_file
from aermanager.parsers import parse_aedat4
from aermanager.preprocess import slice_by_count
from aermanager.preprocess import accumulate_frames
import matplotlib.pyplot as plt
import numpy as np
def save_img(frames,save_dir):
    for x in range(1,len(frames)+1) :
        path = os.path.join(save_dir,str(x) + '.png')
        plt.imsave(path,frames[x-1])

in_dir = r'G:\lxr\gaitRec\dvs_sample'
out_dir = r'G:\lxr\gaitRec\dvs_out'
in_path,out_path = read.list_all_files(in_dir,out_dir)
for i in range(0,len(in_path)):
    shape,events = load_events_from_file(in_path[i],parser=parse_aedat4)
    sliced_events = slice_by_count(events[200000:700000],spike_count=5000)#设定累积数量
    # for j, event in enumerate(sliced_events):
    #     sliced_events[j] = np.array([ev for ev in event if ev[-1]])
    frames = accumulate_frames(sliced_events,bins_y=range(shape[0]+1),bins_x=range(shape[1]+1))
    n, _, w, h = frames.shape
    pf = np.zeros((n, w, h, 3))
    pv = np.array([0,1,0])
    nf = np.zeros((n, w, h, 3))
    nv = np.array([1,0 ,0 ])
    pi = frames[:, 0, :, :] != 0
    ni = frames[:, 1, :, :] != 0
    pf[pi] = pv
    nf[ni] = nv
    frames = (pf + nf)

    save_img(frames,out_path[i])
    print(in_path[i])