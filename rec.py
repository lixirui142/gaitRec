#!/usr/bin/env python
# pylint: disable=W0201
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

from utils import AverageMeter, ExponentialLR

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
from center_loss import CenterLoss

# train set
train_num = 62


class TripletLoss(nn.Module):
	"""Triplet loss with hard positive/negative mining.

	Reference:
		Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

	Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

	Args:
		margin (float, optional): margin for triplet. Default is 0.3.
		prec (int, optional): returned precision type. 0 for rank-all, 1 for rank-1
	"""

	def __init__(self, prec = 0, margin=0.3):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		self.prec = prec

	def forward(self, inputs, targets):
		"""
		Args:
			inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
			targets (torch.LongTensor): ground truth labels with shape (batch_size).
		"""
		n = inputs.size(0)
		# Compute pairwise distance, replace by the official when merged
		dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
		dist = dist + dist.t()
		dist.addmm_(beta=1, alpha=-2, mat1=inputs, mat2=inputs.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

		# For each anchor, find the hardest positive and negative
		mask = targets.expand(n, n).eq(targets.expand(n, n).t())
		dist_ap, dist_an = [], []
		for i in range(n):
			# if mask[i] == 0
			# print(mask[i]==0)
			dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
			dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
		dist_ap = torch.cat(dist_ap)
		dist_an = torch.cat(dist_an)

		# Compute ranking hinge loss
		y = torch.ones_like(dist_an)
		if self.prec:
			srt = dist.argsort(dim=1)
			prec = mask.gather(dim=1, index=srt[:, 1].unsqueeze(dim=1)).sum() * 1. / y.size(0)
		else:
			prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return self.ranking_loss(dist_an, dist_ap, y), prec

	def get_dist(self, x, y):
		m, n = x.size(0), y.size(0)
		x = x.view(m, -1)
		y = y.view(n, -1)
		dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
			   torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
		dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
		return dist

	def test(self, gallery_feat, gallery_lab, probe_feat, probe_lab):
		dist = self.get_dist(probe_feat, gallery_feat)
		m, n = probe_lab.size(0), gallery_lab.size(0)
		# For each anchor, find the hardest positive and negative
		mask = probe_lab.unsqueeze(1).expand(m, n).eq(gallery_lab.unsqueeze(1).expand(n, m).t())
		dist_ap, dist_an = [], []
		for i in range(m):
			# if mask[i] == 0
			# print(mask[i]==0)
			dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
			dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
		dist_ap = torch.cat(dist_ap)
		dist_an = torch.cat(dist_an)

		# Compute ranking hinge loss
		y = torch.ones_like(dist_an)

		srt = dist.argsort(dim=1)
		rank_one = mask.gather(dim=1, index=srt[:, 0].unsqueeze(dim=1)).sum() * 1. / y.size(0)

		#rank_all = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		# return rank_one, rank_all
		return rank_one


# class TripletLoss(nn.Module):
#     """Triplet loss with hard positive/negative mining.

#     Reference:
#         Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

#     Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

#     Args:
#         margin (float, optional): margin for triplet. Default is 0.3.
#     """

#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
#             targets (torch.LongTensor): ground truth labels with shape (num_classes).
#         """
#         n = inputs.size(0)
#         #print(inputs.size())
#         # Compute pairwise distance, replace by the official when merged
#         #dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist=[]
#         for i in range(n):
#             #print(inputs[i,:])
#             anchor=inputs[i,:].expand(inputs.size())
#             #print(anchor,inputs)
#             dist.append(torch.cosine_similarity(anchor,inputs, dim=1))
#         dist=torch.cat(dist)
#         dist = dist.view(n, -1)
#         #print(dist[0,:],dist[1,:],dist[2,:])
#         #print(ere)
#         # dist = dist + dist.t()
#         # dist.addmm_(1, -2, inputs, inputs.t())
#         # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             # dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#             dist_ap.append(dist[i][mask[i]].min().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].max().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)

#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)
#         return self.ranking_loss(dist_an, dist_ap, y)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv1d') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if m.bias is not None:
			m.bias.data.fill_(0)
	elif classname.find('Conv2d') != -1:
		m.weight.data.normal_(0.0, 0.02)
		if m.bias is not None:
			m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


# class REC_IO:
# 	def load_model(self, model, **kwargs):
# 		model = Model(3, graph_args, True, device)


class REC_Processor:
	"""
		Processor for Skeleton-based Action Recgnition
	"""

	def __init__(self, args, data_loader, device):
		self.arg = args
		self.data_loader = data_loader
		self.dev = device
		self.prec_meter = AverageMeter()
		self.loss_meter = AverageMeter()
		self.center_loss = CenterLoss(num_classes=args.train_len, feat_dim=256, use_gpu=True)
		self.with_center_loss = args.enable_center

	def load_model(self, model, pretrain = False):
		self.model = model
		if not pretrain:
			self.model.apply(weights_init)

	def load_optimizer(self):
		if self.arg.optimizer == 'SGD':
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr=self.arg.base_lr,
				momentum=0.9,
				nesterov=self.arg.nesterov,
				weight_decay=self.arg.weight_decay)
		elif self.arg.optimizer == 'Adam':
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.arg.base_lr,
				weight_decay=self.arg.weight_decay)
		else:
			raise ValueError()
		self.scheduler = ExponentialLR(optimizer=self.optimizer, decay_epochs=self.arg.decay_epoch, gamma= self.arg.gamma)
		if self.with_center_loss:
			self.center_optimizer = optim.SGD(
				self.center_loss.parameters(),
				lr = self.arg.center_lr)


	def adjust_lr(self):
		self.scheduler.step()

	# def cosine_decay(self, alpha, epoch, decay_alpha = 0):
	# 	global_step = max(self.arg.rampdown_epoch - epoch, 0)
	# 	cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.arg.rampdown_epoch))
	# 	decayed = (1 - decay_alpha) * cosine_decay + decay_alpha
	# 	decayed_alpha = alpha * decayed
	# 	return decayed_alpha

	# def adjust_alpha(self, alpha, epoch, enable_center):
	# 	if epoch >= self.arg.center_startep and enable_center:
	# 		self.with_center_loss = True
	# 	else:
	# 		self.with_center_loss = False

	def standarization(self):
		if os.path.exists(self.arg.save_dir + '/mean.pkl'):
			with open(self.arg.save_dir + '/mean.pkl', 'rb') as f:
				self.mean = pickle.load(f)
			with open(self.arg.save_dir + '/std.pkl', 'rb') as f:
				self.std = pickle.load(f)
			return
		loader = self.data_loader['init']
		sum = torch.zeros(3)
		n = 0
		for data, label in loader:
			data = data.squeeze(axis=0).float()
			sum += data.sum(axis=(0,2,3,4))
			n += data.numel() // 3

		mean = sum / n
		std = torch.zeros(3)
		for data, label in loader:
			data = data.squeeze(axis=0).float()
			std += (((data.permute(0,2,3,4,1) - mean) ** 2).sum(axis=(0,1,2,3)) / n)
		std = torch.sqrt(std)
		self.mean = mean
		self.std = std
		with open(self.arg.save_dir + '/mean.pkl', 'wb') as f:
			pickle.dump(self.mean, f)
		with open(self.arg.save_dir + '/std.pkl', 'wb') as f:
			pickle.dump(self.std, f)

	def train(self):
		self.model.train()
		loader = self.data_loader['train']
		loss_value = []
		prec_value = []
		i = 0
		print_itv = 10
		alpha = self.arg.alpha
		trip = TripletLoss()
		flag = 'mean' in dir(self)
		for data, label in loader:
			if not flag:
				data = data.squeeze(axis=0).float().to(self.dev)
			else:
				data = data.squeeze(axis=0).float()
				data = ((data.permute(0, 2, 3, 4, 1) - self.mean) / self.std).permute(0, 4, 1, 2, 3).to(self.dev)
			label = label.long().to(self.dev)
			output = self.model(data)
			# self.show(output,10)
			loss3, prec = trip.forward(output, label)
			if self.with_center_loss:
				c_loss = self.center_loss.forward(output, label)
			else:
				# c_output = output.detach()
				# c_loss = self.center_loss.forward(c_output, label)
				c_loss = 0

			loss = loss3 + alpha * c_loss

			self.optimizer.zero_grad()
			if self.with_center_loss:
				self.center_optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if self.with_center_loss:
				self.center_optimizer.step()
			self.prec_meter.update(prec)
			self.loss_meter.update(loss.data.item())

			if i % print_itv == (print_itv - 1):  # print every 2000 mini-batches
				print('[%5d] loss: %.5f.' %
					  (i + 1, self.loss_meter.avg), 'prec : %.5f.' %
					  self.prec_meter.avg)
				loss_value.append(self.loss_meter.avg)
				prec_value.append(self.prec_meter.avg)
			i += 1
		return loss_value, prec_value
	# self.epoch_info['mean_loss'] = np.mean(loss_value)
	# self.show_epoch_info()
	# self.io.print_timer()

	def test(self, evaluation=True, only_nm = False):

		self.model.eval()
		loader = self.data_loader['test']
		gallary_feat = []
		gallary_lab = []
		trip = TripletLoss(prec = 1)
		loader.dataset.set_mode(0)
		flag = 'mean' in dir(self)
		for data, label in loader:
			# get data
			if not flag:
				data = data.float().to(self.dev)
			else:
				data = data.float()
				data = ((data.permute(0, 2, 3, 4, 1) - self.mean) / self.std).permute(0, 4, 1, 2, 3).to(self.dev)
			label = label.long().to(self.dev)
			with torch.no_grad():
				output = self.model(data)
			gallary_feat.append(output)
			gallary_lab.append(label)
		gallary_feat = torch.cat(gallary_feat)
		gallary_lab = torch.cat(gallary_lab).long()
		rank_one = []
		if only_nm:
			mxr = 2
		else:
			mxr = 4
		for i in range(1,mxr):
			loader.dataset.set_mode(i)
			probe_feat = []
			probe_lab = []
			for data, label in loader:
				# get data
				if not flag:
					data = data.float().to(self.dev)
				else:
					data = data.float()
					data = ((data.permute(0, 2, 3, 4, 1) - self.mean) / self.std).permute(0, 4, 1, 2, 3).to(self.dev)
				label = label.long().to(self.dev)
				with torch.no_grad():
					output = self.model(data)
				probe_feat.append(output)
				probe_lab.append(label)
			probe_feat = torch.cat(probe_feat)
			probe_lab = torch.cat(probe_lab).long()
			with torch.no_grad():
				rank_one.append(trip.test(gallary_feat, gallary_lab, probe_feat, probe_lab))
		if only_nm:
			rank_one += [0, 0]
			avg = sum(rank_one) / 1
			print("Test Result: nm: {:.5f}.".format(rank_one[0]))
			return rank_one, avg
		else:
			avg = sum(rank_one) / 3
			print("Test Resut: nm: {:.5f}. bg: {:.5f}. cl: {:.5f}. Total: {:.5f}".format(rank_one[0], rank_one[1], rank_one[2], avg))
			return rank_one, avg

	def show(self, output3, step):
		temp = []
		temp1 = []
		temp2 = []
		temp3 = []
		# temp4 = []
		# temp5 = []
		output3 = output3.data.cpu().numpy()
		output3 = list(output3)
		idd = 0
		for decompos in output3:
			print(len(decompos))

			if int(idd / step) == 0:
				temp.extend(decompos)
			if int(idd / step) == 1:
				temp1.extend(decompos)
			if int(idd / step) == 2:
				temp2.extend(decompos)
			if int(idd / step) == 3:
				temp3.extend(decompos)
			# if int(idd / step) == 4:
			# 	temp4.extend(decompos)
			# if int(idd / step) == 5:
			# 	temp5.extend(decompos)
			idd += 1
		# print(len(temp))

		temp111 = []
		temp111.extend(temp)
		temp111.extend(temp1)
		temp111.extend(temp2)
		temp111.extend(temp3)
		# temp111.extend(temp4)
		# temp111.extend(temp5)

		temp111 = np.array(temp111)
		length = 256
		# length=3*100*18
		temp111 = temp111.reshape(-1, length)
		print(temp111.shape)

		temp = np.array(temp)
		temp1 = np.array(temp1)
		temp2 = np.array(temp2)
		temp3 = np.array(temp3)
		# temp4 = np.array(temp4)
		# temp5 = np.array(temp5)

		temp = temp.reshape(-1, length)
		temp1 = temp1.reshape(-1, length)
		temp2 = temp2.reshape(-1, length)
		temp3 = temp3.reshape(-1, length)
		# temp4 = temp4.reshape(-1, length)
		# temp5 = temp5.reshape(-1, length)

		fig = plt.figure()
		pca = PCA(n_components=3)
		pca.fit(temp111)
		print(pca.explained_variance_ratio_)
		print(pca.explained_variance_)
		ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

		X = pca.transform(temp)
		# print(X.shape)
		ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='*', c='red', s=40)
		# ax.scatter(X[:, 0],X[:, 1].reshape(1.-1),X[:, 2].reshape(1.-1), marker='*',c='red',s=30)

		X = pca.transform(temp1)
		ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='g')

		X = pca.transform(temp2)
		ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='b')

		X = pca.transform(temp3)
		ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='y')

		# X = pca.transform(temp4)
		# ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='c')
		#
		# X = pca.transform(temp5)
		# ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='k')
		# codebook=center_point[i]
		# plt.scatter(codebook[:, 0], codebook[:, 1],codebook[:, 2],c='red',marker='o',linewidths=5)
		plt.show()

	# print(ere)


	@staticmethod
	def get_parser(add_help=False):
		parser = argparse.ArgumentParser(
			add_help=add_help,
			description='Spatial Temporal Graph Convolution Network')

		parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
							help='which Top K accuracy will be shown')
		parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
		parser.add_argument('--step', type=int, default=[], nargs='+',
							help='the epoch where optimizer reduce the learning rate')
		parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
		parser.add_argument('--nesterov', type=bool, default=True, help='use nesterov or not')
		parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

		return parser

# python main.py recognition --phase train -c /home/bird/xuke/st-gcn-master/config/st_gcn/casia-skeleton/train.yaml
# python main.py recognition  --phase test -c /home/bird/xuke/st-gcn-master/config/st_gcn/casia-skeleton/test.yaml


	# def cosine_decay(global_step):
	# 	global_step = min(global_step, args.rampdown_epoch)
	# 	cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / args.rampdown_epoch))
	# 	decayed = (1 - args.decay_alpha) * cosine_decay + args.decay_alpha
	# 	decayed_learning_rate = args.lr * decayed
	# 	return decayed_learning_rate

	# def natural_decay(step):
	# 	decayed_learning_rate = args.lr * np.exp(-args.decay_rate * step / args.rampdown_epoch)
	# 	return decayed_learning_rate

	# def cosine_rampdown(current, rampdown_length):
	#     assert 0 <= current <= rampdown_length
	#     return max(0., float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)))
	# def extract2(self, data, label):

	# 	# xf=data[:,:,0:50,:,:]
	# 	# xb=data[:,:,50:100,:,:]

	# 	# bat=xf.size(0)
	# 	# bat1=xf.size()
	# 	# xf = xf.reshape(bat, -1)
	# 	# mean = xf.mean(dim=1).reshape(bat,-1)
	# 	# std = xf.std(dim=1, unbiased=False).reshape(bat,-1)
	# 	# xf = (xf - mean)/(std+1e-5)
	# 	# xf=xf.reshape(bat1)

	# 	# bat=xb.size(0)
	# 	# bat1=xb.size()
	# 	# xb = xb.reshape(bat, -1)
	# 	# mean = xb.mean(dim=1).reshape(bat,-1)
	# 	# std = xb.std(dim=1, unbiased=False).reshape(bat,-1)
	# 	# xb = (xb - mean)/(std+1e-5)
	# 	# xb=xb.reshape(bat1)

	# 	# data=torch.cat((xf,xb),1)
	# 	# 100,3,100,18,1

	# 	data = data.contiguous().view(data.size(0), -1)
	# 	# label = label.contiguous().view(label.size(0), -1)

	# 	a = data[label.gt(0)]
	# 	label = label * -1 + 1
	# 	b = data[label.gt(0)]

	# 	data = data.data.cpu().numpy()
	# 	a = a.data.cpu().numpy()
	# 	b = b.data.cpu().numpy()

	# 	print(data.shape)

	# 	fig = plt.figure()
	# 	pca = PCA(n_components=3)
	# 	pca.fit(data)
	# 	# print pca.explained_variance_ratio_
	# 	# print pca.explained_variance_
	# 	ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
	# 	X = pca.transform(a)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='*', c='red', s=40)
	# 	# ax.scatter(X[:, 0],X[:, 1].reshape(1.-1),X[:, 2].reshape(1.-1), marker='*',c='red',s=30)
	# 	X = pca.transform(b)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='g')
	# 	plt.show()

	# # print(ere)

	# def extract(self, xf, xb):
	# 	# print(data.size())
	# 	# print(data[0,:,0,1,0])
	# 	# #print(ere)0
	# 	# xf=data[:,:,0:50,:,:]
	# 	# xb=data[:,:,50:100,:,:]

	# 	# #xf=data
	# 	xf = xf.contiguous().view(xf.size(0), -1)
	# 	xb = xb.contiguous().view(xb.size(0), -1)
	# 	# bat=xf.size(0)
	# 	# bat1=xf.size()
	# 	# xf = xf.reshape(bat, -1)
	# 	# mean = xf.mean(dim=1).reshape(bat,-1)
	# 	# std = xf.std(dim=1, unbiased=False).reshape(bat,-1)
	# 	# xf = (xf - mean)/(std+1e-5)
	# 	# xf=xf.reshape(bat1)

	# 	# print(xf,xb)
	# 	# print(ere)
	# 	# print(ere)
	# 	# print(xf[:,:6],xb[:,:6])

	# 	# mean = xf.mean(dim=0)
	# 	# std = xf.std(dim=0, unbiased=False)
	# 	# print(mean,std)
	# 	# xf = (xf - mean)/(std+1e-5)

	# 	xf = xf.data.cpu().numpy()
	# 	# xf=xf.astype(np.float)
	# 	# print(ere)
	# 	kk = 10
	# 	temp = xf[0:1 * kk, :]
	# 	print(temp)
	# 	# temp=np.vstack((temp,recon_y1[0:1*kk,:]))
	# 	temp1 = xf[1 * kk:2 * kk, :]
	# 	# print(temp1)
	# 	# temp1=np.vstack((temp1,recon_y1[1*kk:2*kk,:]))
	# 	temp2 = xf[2 * kk:3 * kk, :]
	# 	# print(temp2)
	# 	# temp2=np.vstack((temp2,recon_y1[2*kk:3*kk,:]))
	# 	temp3 = xf[3 * kk:4 * kk, :]
	# 	# print(temp3)
	# 	# temp3=np.vstack((temp3,recon_y1[3*kk:4*kk,:]))
	# 	temp4 = xf[4 * kk:5 * kk, :]
	# 	print(temp4)
	# 	# temp4=np.vstack((temp4,recon_y1[4*kk:5*kk,:]))
	# 	temp5 = xf[5 * kk:6 * kk, :]
	# 	# print(temp5)
	# 	temp6 = xf[6 * kk:7 * kk, :]
	# 	# temp5=np.vstack((temp5,recon_y1[5*kk:6*kk,:]))
	# 	# print(temp4,temp5)

	# 	temp111 = temp
	# 	temp111 = np.vstack((temp111, temp1))
	# 	temp111 = np.vstack((temp111, temp2))
	# 	temp111 = np.vstack((temp111, temp3))
	# 	temp111 = np.vstack((temp111, temp4))
	# 	temp111 = np.vstack((temp111, temp5))
	# 	temp111 = np.vstack((temp111, temp6))
	# 	print(temp111.shape)

	# 	fig = plt.figure()
	# 	pca = PCA(n_components=3)
	# 	pca.fit(temp111)
	# 	# print pca.explained_variance_ratio_
	# 	# print pca.explained_variance_
	# 	ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

	# 	X = pca.transform(temp)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='*', c='red', s=40)
	# 	# ax.scatter(X[:, 0],X[:, 1].reshape(1.-1),X[:, 2].reshape(1.-1), marker='*',c='red',s=30)

	# 	X = pca.transform(temp1)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='g')

	# 	X = pca.transform(temp2)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='b')

	# 	X = pca.transform(temp3)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='y')

	# 	X = pca.transform(temp4)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='c')

	# 	X = pca.transform(temp5)
	# 	# print(X)
	# 	ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=40, marker='*', c='k')

	# 	# X = pca.transform(temp6)
	# 	# #print(X)
	# 	# ax.scatter(X[:, 0],X[:, 1],X[:, 2],s=40, marker='*',c='m')
	# 	# codebook=center_point[i]
	# 	# plt.scatter(codebook[:, 0], codebook[:, 1],codebook[:, 2],c='red',marker='o',linewidths=5)
	# 	plt.show()

	# # print(ere)