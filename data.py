import torch
import os
import random
import json
import numpy as np
import itertools
from torch.utils.data.sampler import Sampler
import torch
import re
from torch._six import container_abcs, string_classes, int_classes


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
		for i in range(clip_len):
			if i < len(files):
				f = open(os.path.join(root, files[i]), 'r')
				str = f.read()
				res.append(json.loads(str)['people'][0]['pose_keypoints_2d'])
			else:
				res.append([0.0 for i in range(len(res[-1]))])
	x = torch.tensor(res)
	x = x.view(len(res), -1, 3, 1)
	x = x.permute(2, 0, 1, 3).contiguous()
	return x


class GaitDataset(torch.utils.data.Dataset):
	def __init__(self, args):
		self.args = args
		self.res = get_files(self.args.data_dir, viewset=self.args.viewset)
		self.samples_len = len(self.args.sample_list)
		self.samples = []

	def __len__(self):
		return len(self.samples)

class InitDataset(GaitDataset):
	def __init__(self, args):
		super(InitDataset, self).__init__(args)
		self.train_len = len(self.args.id_list)
		for identity in self.args.id_list:
			for idx in self.args.sample_list:
				self.samples.append(self.res[identity][idx])

	def __getitem__(self, index):
		x = self.samples[index][0]
		clip_len = min(self.samples[index][1], self.args.clip_len)
		# clip_len = 60
		label = int(x.split('-')[-4][-3:]) - 1
		x = get_content(x, clip_len)

		return x, label


class TrainDataset(GaitDataset):
	def __init__(self, args):
		super(TrainDataset, self).__init__(args)
		self.train_len = len(self.args.id_list)
		for identity in self.args.id_list:
			for idx in self.args.sample_list:
				self.samples.append(self.res[identity][idx])

	def __getitem__(self, index):
		# x = self.samples[index][0]
		# clip_len = min(self.samples[index][1], self.args.clip_len)
		# label = int(x.split('-')[-4][-3:]) - 1
		# x = get_content(x, clip_len)
		clip_len = min(self.samples[index][1], self.args.clip_len)
		cls = index // self.samples_len
		idx = index % self.samples_len
		lst = []

		cls_list = [cls]
		idx_list = [idx]
		label = []
		while (len(cls_list) <= self.args.batch_class_num):
			cls = cls_list[-1]
			while (len(idx_list) <= self.args.class_sample_num):
				smpidx = idx_list[-1]
				smp = self.samples[cls * self.samples_len + smpidx]
				lst.append(smp[0])
				clip_len = min(clip_len, smp[1])
				label.append(cls)

				smpidx = random.randint(0, self.samples_len - 1)
				while (smpidx in idx_list):
					smpidx = random.randint(0, self.samples_len - 1)
				idx_list.append(smpidx)

			smpcls = random.randint(0, self.train_len - 1)
			while (smpcls in cls_list):
				smpcls = random.randint(0, self.train_len - 1)
			cls_list.append(smpcls)
			idx_list = [random.randint(0, self.samples_len - 1)]

		label = torch.tensor(label)

		data = []
		for f in lst:
			data.append(get_content(f, clip_len))
		data = torch.stack(data, dim=0)

		return data, label

	def __len__(self):
		return len(self.samples)


class TestDataset(GaitDataset):
	def __init__(self, args):
		super(TestDataset, self).__init__(args)
		self.train_len = len(self.args.id_list)
		self.test_len = len(self.args.test_list)
		self.probe_nm = []
		self.probe_bg = []
		self.probe_cl = []
		self.gallary = []
		for identity in self.args.test_list:
			for idx in self.args.gallary_sample:
				self.gallary.append(self.res[identity][idx])
			for idx in self.args.probe_nm_sample:
				self.probe_nm.append(self.res[identity][idx])
			for idx in self.args.probe_bg_sample:
				self.probe_bg.append(self.res[identity][idx])
			for idx in self.args.probe_cl_sample:
				self.probe_cl.append(self.res[identity][idx])
		self.mode = False
		self.samples = self.gallary

	def set_mode(self, mode = 0):
		self.mode = mode
		if self.mode == 0:
			self.samples = self.gallary
		elif self.mode == 1:
			self.samples = self.probe_nm
		elif self.mode == 2:
			self.samples = self.probe_bg
		elif self.mode == 3:
			self.samples = self.probe_cl

	def __getitem__(self, index):
		x = self.samples[index][0]
		clip_len = min(self.samples[index][1], self.args.clip_len)
		# clip_len = 60
		label = int(x.split('-')[-4][-3:]) - 1
		x = get_content(x, clip_len)

		return x, label

	# clip_len = min(self.samples[index][1], self.args.clip_len)
	# cls = index // self.samples_len
	# idx = index % self.samples_len
	# lst = []
	#
	# cls_list = [cls]
	# idx_list = [idx]
	# label = []
	# while (len(cls_list) <= self.args.batch_class_num):
	# 	cls = cls_list[-1]
	# 	while (len(idx_list) <= self.args.class_sample_num):
	# 		smpidx = idx_list[-1]
	# 		smp = self.samples[cls * self.samples_len + smpidx]
	# 		lst.append(smp[0])
	# 		clip_len = min(clip_len, smp[1])
	# 		label.append(cls + self.train_len)
	#
	# 		smpidx = random.randint(0, self.samples_len - 1)
	# 		while (smpidx in idx_list):
	# 			smpidx = random.randint(0, self.samples_len - 1)
	# 		idx_list.append(smpidx)
	#
	# 	smpcls = random.randint(0, self.train_len - 1)
	# 	while (smpcls in cls_list):
	# 		smpcls = random.randint(0, self.train_len - 1)
	# 	cls_list.append(smpcls)
	# 	idx_list = [random.randint(0, self.samples_len - 1)]
	#
	# label = torch.tensor(label)
	#
	# data = []
	# for f in lst:
	# 	data.append(get_content(f, clip_len))
	# data = torch.stack(data, dim=0)
	#
	# return data, label

	def __len__(self):
		return len(self.samples)

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def rec_collate(batch):
	r"""Puts each data field into a tensor with outer dimension batch size"""

	elem = batch[0]
	elem_type = type(elem)
	if isinstance(elem, torch.Tensor):
		out = None
		if torch.utils.data.get_worker_info() is not None:
			# If we're in a background process, concatenate directly into a
			# shared memory tensor to avoid an extra copy
			numel = sum([x.numel() for x in batch])
			storage = elem.storage()._new_shared(numel)
			out = elem.new(storage)
		clip_len = min([data.shape[1] for data in batch])
		batch = [data[:,:clip_len] for data in batch]
		return torch.stack(batch, 0, out=out)
	elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
			and elem_type.__name__ != 'string_':
		elem = batch[0]
		if elem_type.__name__ == 'ndarray':
			# array of string classes and object
			if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
				raise TypeError(default_collate_err_msg_format.format(elem.dtype))

			return rec_collate([torch.as_tensor(b) for b in batch])
		elif elem.shape == ():  # scalars
			return torch.as_tensor(batch)
	elif isinstance(elem, float):
		return torch.tensor(batch, dtype=torch.float64)
	elif isinstance(elem, int_classes):
		return torch.tensor(batch)
	elif isinstance(elem, string_classes):
		return batch
	elif isinstance(elem, container_abcs.Mapping):
		return {key: rec_collate([d[key] for d in batch]) for key in elem}
	elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
		return elem_type(*(rec_collate(samples) for samples in zip(*batch)))
	elif isinstance(elem, container_abcs.Sequence):
		# check to make sure that the elements in batch have consistent size
		it = iter(batch)
		elem_size = len(next(it))
		if not all(len(elem) == elem_size for elem in it):
			raise RuntimeError('each element in list of batch should be of equal size')
		transposed = zip(*batch)
		return [rec_collate(samples) for samples in transposed]

	raise TypeError(default_collate_err_msg_format.format(elem_type))

# class TwoStreamBatchSampler(Sampler):
# 	"""Iterate two sets of indices
#
# 	An 'epoch' is one iteration through the primary indices.
# 	During the epoch, the secondary indices are iterated through
# 	as many times as needed.
# 	"""
#
# 	def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, unlabeled_size_limit=None):
# 		self.primary_indices = primary_indices
# 		self.secondary_indices = secondary_indices
# 		self.secondary_batch_size = secondary_batch_size
# 		self.primary_batch_size = batch_size - secondary_batch_size
# 		self.unlabeled_size_limit = unlabeled_size_limit
#
# 		assert len(self.primary_indices) >= self.primary_batch_size > 0
# 		assert len(self.secondary_indices) >= self.secondary_batch_size > 0
#
# 	def __iter__(self):
# 		primary_iter = iterate_once(self.primary_indices, self.unlabeled_size_limit)
# 		secondary_iter = iterate_eternally(self.secondary_indices)
# 		return (
# 			primary_batch + secondary_batch
# 			for (primary_batch, secondary_batch)
# 			in zip(grouper(primary_iter, self.primary_batch_size),
# 				   grouper(secondary_iter, self.secondary_batch_size))
# 		)
#
# 	def __len__(self):
# 		if self.unlabeled_size_limit is None:
# 			return len(self.primary_indices) // self.primary_batch_size
# 		else:
# 			return self.unlabeled_size_limit // self.primary_batch_size
#
#
# def iterate_once(iterable, unlabeled_size_limit=None):
# 	if unlabeled_size_limit is None:
# 		return np.random.permutation(iterable)
# 	else:
# 		result = np.random.permutation(iterable)[:unlabeled_size_limit]
# 		return result
#
#
# def iterate_eternally(indices):
# 	def infinite_shuffles():
# 		while True:
# 			yield np.random.permutation(indices)
#
# 	return itertools.chain.from_iterable(infinite_shuffles())
#
#
# def grouper(iterable, n):
# 	"Collect data into fixed-length chunks or blocks"
# 	args = [iter(iterable)] * n
# 	return zip(*args)
