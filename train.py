from torch.nn.functional import dropout
from net.st_gcn import Model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import TrainDataset, TestDataset, rec_collate, InitDataset
from utils import plot, plotmulti, ExponentialLR
from rec import REC_Processor
import pickle
import numpy as np
import wandb
from args import get_args


def main():
	args = get_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	graph_args = {"layout": 'openpose', "strategy": 'uniform', "max_hop": 1, "dilation": 1}
	model = Model(3, graph_args, True, device)
	model = model.to(device)

	# parser = REC_Processor.get_parser()
	# args = parser.parse_args()

	# # args.data_dir = "G:/program/github/gaitRec/data/prime_joints/"
	# # args.data_dir="G:/xuke/CASIA-B/prime-joints/"
	# args.data_dir = "G:/lxr/gaitRec/data/prime_joints/"
	# # args.data_dir="C:/Users/admn/Desktop/gait prp/gait-gcn/data/prime-joints/"
	# ## 001-bg-01-000
	# args.threads = 1
	# args.batchSize = 64
	# args.lr = 0.1
	# args.epoch = 200
	# args.gamma = 0.01
	# args.decay_epoch = args.epoch
	# args.name = "test_eb"
	# args.save_dir = "model/" + args.name
	# args.result_dir = "result/" + args.name
	# # args.viewset = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
	# args.viewset = ["090"]
	# args.clip_len = 30
	# # args.sample_list = [i for i in range(10 * len(args.viewset))]
	# # args.probe_nm_sample = [8, 9]
	# # args.probe_bg_sample = [0, 1]
	# # args.probe_cl_sample = [2, 3]
	# # args.gallary_sample = [4, 5, 6, 7]
	# args.probe_bg_sample = []
	# args.sample_list = [i for i in range(6 * len(args.viewset))]
	# args.probe_nm_sample = [4, 5]
	# args.probe_cl_sample = []
	# args.gallary_sample = [0, 1, 2, 3]
	# args.id_list = [i for i in range(62)]
	# args.train_len = len(args.id_list)
	# args.test_list = [i for i in range(62, 124)]
	# args.load_pretrain = False
	# args.pretrain = "model/090rm2/ckpt_best_0.91398.pth"
	# args.class_rate = 1.0
	# args.evaluate = False
	# args.batch_class_num = 8
	# args.class_sample_num = 4
	# args.save_freq = 10
	# args.alpha = 0.0
	# args.center_lr = 1.0
	# args.center_startep = 25
	# args.enable_center = False
	# args.wandb = True
	# args.only_nm = True

	if args.wandb:
		wandb.init(entity="lixirui142", project="gaitRec",name=args.name)
		wandb.config.update(args)

	if args.load_pretrain:
		ckpt = torch.load(args.pretrain, map_location=torch.device('cpu'))
		model.load_state_dict(ckpt['state_dict'], strict=True)

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	if not os.path.exists(args.result_dir):
		os.mkdir(args.result_dir)

	init_set = InitDataset(args)

	init_loader = DataLoader(dataset=init_set, num_workers=args.threads,
							batch_size=args.batch_class_num * args.class_sample_num, shuffle=True, collate_fn=rec_collate)

	train_set = TrainDataset(args)

	train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=1, shuffle=True)

	test_set = TestDataset(args)

	test_loader = DataLoader(dataset=test_set, num_workers=args.threads,
							batch_size=args.batch_class_num * args.class_sample_num, shuffle=True, collate_fn=rec_collate)

	data_loader = {"train": train_loader, "test": test_loader, "init": init_loader}

	proc = REC_Processor(args, data_loader, device)
	proc.load_model(model, pretrain=args.pretrain)
	proc.load_optimizer()

	# test_set = TestDataset(args)
	# test_indices = [i for i in range(len(test_set))]
	# test_sampler = TwoStreamBatchSampler(test_indices, test_indices, args.batchSize, args.batchSize - 1)
	# test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size = 1)

	# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
	# criterion = nn.MSELoss()

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

	# def adjust_learning_rate(current):
	#     lr = args.lr
	#     lr *= cosine_rampdown(current, args.rampdown_epoch)
	#     return lr
	#proc.standarization()
	loss = []
	prec = []
	test_rank1 = [[], [], []]
	best = 0
	rank_one, avg = proc.test(only_nm= args.only_nm)
	bestrankone = rank_one
	best = avg
	for i in range(3):
		test_rank1[i].append(rank_one[i])
	if args.wandb:
		wandb.log({"NM": rank_one[0], "BG": rank_one[1], "CL": rank_one[2], "AVG": avg})

	for epoch in range(args.epoch):
		#proc.adjust_alpha(args.alpha, epoch, args.enable_center)
		print("Epoch %d" % epoch)
		tloss, tprec = proc.train()
		proc.adjust_lr()
		loss += tloss
		prec += tprec
		print("Test: ")
		rank_one, avg = proc.test(only_nm= args.only_nm)
		for i in range(3):
			test_rank1[i].append(rank_one[i])
		if args.wandb:
			wandb.log({"NM": rank_one[0], "BG": rank_one[1], "CL": rank_one[2], "AVG": avg})
		if avg > best:
			best = avg
			bestrankone = rank_one
			if not os.path.exists(args.save_dir):
				os.makedirs(args.save_dir)
			model_out_path = "{}/ckpt_best_{:.5f}.pth".format(args.save_dir, best)
			torch.save({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
			}, model_out_path)
		print("Max acu: nm {:.5f}, bg {:.5f}, cl {:.5f}".format(bestrankone[0], bestrankone[1], bestrankone[2]))
		print("Max total: {:.5f}. Max nm: {:.5f}. Max bg: {:.5f}. Max cl: {:.5f}"
			  .format(best, max(test_rank1[0]), max(test_rank1[1]), max(test_rank1[2])))

		if not args.wandb and (epoch + 1) % args.save_freq == 0:
			plot(loss, 'loss', epoch, 'Every 10 Batch', 'Running Loss', args.result_dir)
			plot(prec, 'prec', epoch, 'Every 10 Batch', 'Running Prec', args.result_dir)
			plotmulti(test_rank1, 'rank1prec', epoch, 'Every epoch', 'Test Rank 1 Precision', ['nm', 'bg', 'cl'],
					  args.result_dir)
	else:
		epoch = args.epoch
		if not args.wandb:
			plot(loss, 'loss', epoch, 'Every 10 Batch', 'Running Loss', args.result_dir)
			plot(prec, 'prec', epoch, 'Every 10 Batch', 'Running Prec', args.result_dir)
			plotmulti(test_rank1, 'rank1prec', epoch, 'Every epoch', 'Test Rank 1 Precision', ['nm', 'bg', 'cl'],
					args.result_dir)
		else:
			wandb.config.best_avg = best
			wandb.config.best_nm = bestrankone[0]
			wandb.config.best_bg = bestrankone[1]
			wandb.config.best_cl = bestrankone[2]

		with open(args.save_dir + '/loss.pkl', 'wb') as f:
			pickle.dump(loss, f)
		with open(args.save_dir + '/prec.pkl', 'wb') as f:
			pickle.dump(prec, f)
		with open(args.save_dir + '/test_rank1.pkl', 'wb') as f:
			pickle.dump(test_rank1, f)


if __name__ == '__main__':
	main()