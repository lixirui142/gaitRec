import torch
import os
import random
import json
import numpy as np
import itertools
from torch.utils.data.sampler import Sampler


def get_files(mydir, viewset):
    f = open("len.json", "r")
    lendic = json.load(f)

    res = []
    for root, dirs, files in os.walk(mydir, followlinks=True):
        dirs.sort(key=lambda x: x.split('-')[:2])
        previous = dirs[0].split('-')[0]
        m = []
        for f in dirs:
            t = f.split('-')
            if t[0] != previous:
                while (len(m) < 10 * len(viewset)):
                    m.append(m[-1])
                res.append(m)
                m = []
                previous = t[0]
            if t[-1] in viewset:
                m.append([os.path.join(root, f), lendic[f]])
        res.append(m)
        break
    # print (len(res),len(res[0]))
    # print (res[0])
    return res


def get_content(filedir, clip_len):
    res = []
    for root, dirs, files in os.walk(filedir, followlinks=True):
        files.sort(key=lambda x: x.split('_')[1])
        for f in files[:clip_len]:
            f = open(os.path.join(root, f), 'r')
            str = f.read()
            res.append(json.loads(str)['people'][0]['pose_keypoints_2d'])
    x = torch.tensor(res)
    x = x.view(len(res), -1, 3, 1)
    x = x.permute(2, 0, 1, 3).contiguous()
    return x


class GaitDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.args = args
        self.res = get_files(self.args.data_dir, viewset=self.args.viewset)
        self.samples_len = len(self.args.sample_list)
        self.train_len = len(self.args.id_list)
        self.samples = []
        for identity in self.args.id_list:
            for idx in self.args.sample_list:
                self.samples.append(self.res[identity][idx])

    def __getitem__(self, index):
        x = self.samples[index][0]
        clip_len = min(self.samples[index][1], self.args.clip_len)
        label = int(x.split('-')[-4][-3:]) - 1
        x = get_content(x, clip_len)

        return x, label

    def __len__(self):
        return len(self.samples)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.args = args
        self.res = get_files(self.args.data_dir, viewset=self.args.viewset)
        self.samples_len = len(self.args.sample_list)
        self.test_len = len(self.args.test_list)
        self.samples = []
        for identity in self.args.test_list:
            for idx in self.args.sample_list:
                self.samples.append(self.res[identity][idx])

                # lst = [s1]
                #
                # for s2 in id[4:]:
                #     lst.append((s2))
                # for j in range(self.args.train_length):
                #     if j != k:
                #         lst.append(res[j][i + 4])
                # self.samples.append(lst)

    def __getitem__(self, index):
        # x = self.samples[index][0]
        # clip_len = min(self.samples[index][1], self.args.clip_len)
        # label = int(x.split('-')[-4][-3:]) - 1
        # x = get_content(x, clip_len)
        #
        # return x, label
        x = self.samples[index][0]
        clip_len = min(self.samples[index][1], self.args.clip_len)
        k = index // self.samples_len
        idx = index % self.samples_len
        lst = [x]

        sample_list = set()

        for j in range(1):
            t = random.randint(0, self.samples_len - 1)
            while (t == idx):
                t = random.randint(0, self.samples_len - 1)
            s2 = self.samples[k * self.samples_len + t]
            sample_list.add(t)
            lst.append(s2[0])
            clip_len = min(clip_len, s2[1])

        label = torch.tensor([1.0 for i in range(2)] + [0.0 for j in range(self.test_len - 1)])

        sample_list = set()

        for j in range(self.test_len):
            t = random.randint(0, self.samples_len - 1)
            if (j == k):
                continue

            t = j * self.samples_len + t

            s2 = self.res[self.args.test_list[t // self.samples_len]][self.args.sample_list[t % self.samples_len]]
            lst.append(s2[0])
            clip_len = min(clip_len, s2[1])

        data = []
        for f in lst:
            data.append(get_content(f, clip_len))
        data = torch.stack(data, dim=0)
        return data, label

    def __len__(self):
        return len(self.samples)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, unlabeled_size_limit=None):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.unlabeled_size_limit = unlabeled_size_limit

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices, self.unlabeled_size_limit)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        if self.unlabeled_size_limit is None:
            return len(self.primary_indices) // self.primary_batch_size
        else:
            return self.unlabeled_size_limit // self.primary_batch_size


def iterate_once(iterable, unlabeled_size_limit=None):
    if unlabeled_size_limit is None:
        return np.random.permutation(iterable)
    else:
        result = np.random.permutation(iterable)[:unlabeled_size_limit]
        return result


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)
