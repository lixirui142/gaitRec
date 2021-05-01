import os
import glob
import json
import matplotlib.pyplot as plt
import torch

def get_len(mydir):
    dic = dict()
    for root, dirs, files in os.walk(mydir, followlinks=True):
        for i, f in enumerate(dirs):
            files = glob.glob(os.path.join(root, f) + "/*")
            dic[f] = len(files)
            if i % 100 == 99:
                print(f)
        break
    f = open("len.json", "w")
    json.dump(dic, f)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot(data, name, epoch, xlabel, ylabel, plt_dir):
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt_path = "{}/epoch{}_{}.png".format(plt_dir, str(epoch), name)
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plt_path)
    plt.clf()  # Clear figure

def plotmulti(data, name, epoch, xlabel, ylabel, label, plt_dir):
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt_path = "{}/epoch{}_{}.png".format(plt_dir, str(epoch), name)
    fig, ax = plt.subplots()
    for d, l in zip(data,label):
        ax.plot(d, label = l)
    ax.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plt_path)
    # plt.cla()  # Clear axis
    # plt.clf()  # Clear figure
    plt.close(fig)




# def get_files(mydir, viewset):
#   f = open("len.json", "r")
#   lendic = json.load(f)
#
#   res = []
#   for root, dirs, files in os.walk(mydir, followlinks=True):
#       dirs.sort(key=lambda x: x.split('-')[:2])
#       previous = dirs[0].split('-')[0]
#       m = []
#       for f in dirs:
#           t = f.split('-')
#           if t[0] != previous:
#               while (len(m) < 10 * len(viewset)):
#                   m.append(m[-1])
#               res.append(m)
#               m = []
#               previous = t[0]
#           if t[-1] in viewset:
#               m.append(os.path.join(root, f))
#       res.append(m)
#       break
#   # print (len(res),len(res[0]))
#   # print (res[0])
#   return res
#
#
# def proc_content(filedir):
#   res = []
#   for root, dirs, files in os.walk(filedir, followlinks=True):
#       files.sort(key=lambda x: x.split('_')[1])
#       for i in range(len(files)):
#           if i < len(files):
#               f = open(os.path.join(root, files[i]), 'r')
#               str = f.read()
#               res.append(json.loads(str)['people'][0]['pose_keypoints_2d'])
#           else:
#               res.append([0.0 for i in range(len(res[-1]))])
#
# data_dir="G:/program/github/gait-gcn/data/prime-joints/"
#
# res = get_files(data_dir, ["180"])
#
# for identity in res:
#   for idx in identity:
#       proc_content(idx)