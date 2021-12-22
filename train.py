from torch.nn.functional import dropout
from net.st_gcn import Model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import TrainDataset, TestDataset, rec_collate
from utils import plot, plotmulti
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

	train_set = TrainDataset(args)
	train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=1, shuffle=True)

	test_set = TestDataset(args)
	test_loader = DataLoader(dataset=test_set, num_workers=args.threads,
							batch_size=args.batch_class_num * args.class_sample_num, shuffle=True, collate_fn=rec_collate)

	data_loader = {"train": train_loader, "test": test_loader}

	proc = REC_Processor(args, data_loader, device)
	proc.load_model(model, pretrain=args.pretrain)
	proc.load_optimizer()

	#proc.standarization()

	# initialize
	loss = []
	prec = []
	test_rank1 = [[], [], []]
	best = 0
	rank_one, avg = proc.test(only_nm = args.only_nm)
	bestrankone = rank_one
	best = avg
	for i in range(3):
		test_rank1[i].append(rank_one[i])
	
	if args.wandb:
		wandb.log({"NM": rank_one[0], "BG": rank_one[1], "CL": rank_one[2], "AVG": avg})


	# start training
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

		# logging
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
			model_best_path = "{}/ckpt_best.pth".format(args.save_dir)
			torch.save({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
			}, model_best_path)

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