from net.st_gcn import Model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import GaitDataset,TestDataset,TwoStreamBatchSampler
import argparse
import matplotlib.pyplot as plt
import numpy as np
from test_t import test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
graph_args = {"layout": 'openpose',"strategy":'spatial',"max_hop":1,"dilation":1}
model = Model(3,graph_args,True,device)
model = model.to(device)

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
args = parser.parse_args()

args.data_dir="G:/program/github/gait-gcn/data/prime-joints/"
# args.data_dir="G:/xuke/CASIA-B/prime-joints/"
# args.data_dir="C:/Users/admn/Desktop/gait prp/gait-gcn/data/prime-joints/"

args.threads = 1
args.batchSize = 64
args.lr = 0.01
args.epoch = 1000
args.rampdown_epoch = 1200
args.save_dir = "model/"
args.viewset = ["180"]
args.counter_num = 8
args.pos_num = 8
args.clip_len = 60
args.sample_list = [i for i in range(10)]
args.id_list = [i for i in range(70)]
args.train_len = len(args.id_list)
args.test_list = [i for i in range(100,124)]
args.load_pretrain = False
args.pretrain = "model/ckpt_149.pth"
args.class_rate = 1.0
args.evaluate = False
args.batch_class_num = 16
args.class_sample_num = 4

if args.load_pretrain:
    ckpt = torch.load(args.pretrain,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'],strict=True)


train_set = GaitDataset(args)
train_indices = [i for i in range(len(train_set))]
train_sampler = TwoStreamBatchSampler(train_indices, train_indices, args.batchSize, args.batchSize - 1)

train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_sampler=train_sampler)
test_set = TestDataset(args)
test_indices = [i for i in range(len(test_set))]
test_sampler = TwoStreamBatchSampler(test_indices, test_indices, args.batchSize, args.batchSize - 1)
test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size = 1)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = nn.MSELoss()

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return max(0., float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)))

def adjust_learning_rate(current):
    lr = args.lr
    lr *= cosine_rampdown(current, args.rampdown_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    res = []
    acu = []
    print("Training start.")
    if args.evaluate:
        print("Testing")
        acu.append(test(model, 0, test_loader))
    for epoch in range(args.epoch):
        print("Epoch: %d" % (epoch + 1))
        running_loss = 0.0
        running_class_loss = 0.0
        running_pred_loss = 0.0
        adjust_learning_rate(epoch)

        model.train()
        for i, batch in enumerate(train_loader, 0):
            torch.cuda.empty_cache()
            data, label = batch[0].to(device), batch[1]
            optimizer.zero_grad()
            pred = model(data, label)

            gt = torch.tensor([[1.0] if lb==label[0] else [0.0] for lb in label],device=device, requires_grad=False)
            pred_loss = criterion(pred, gt)
            loss = pred_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f.' %
                      (epoch + 1, i + 1, running_loss / 10.0))
                res.append(running_loss / 10.0)
                running_loss = 0.0

        if epoch % 100 == 99:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            model_out_path = "{}/ckpt_{}.pth".format(args.save_dir, str(epoch))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, model_out_path)

        if epoch % 10 == 9:
            model.eval()
            print("Testing")
            acu.append(test(model, epoch, test_loader))
            plt_dir = "result/"
            if not os.path.exists(plt_dir):
                os.makedirs(plt_dir)
            plt_path = "{}/epoch_{}_rl.png".format(plt_dir, str(epoch))
            plt.plot(res)
            plt.ylabel('Running Loss')
            plt.xlabel('Every 10 iterations')
            plt.savefig(plt_path)
            plt.clf()
            plt_path = "{}/epoch_{}_ta.png".format(plt_dir, str(epoch))
            plt.plot(acu)
            plt.ylabel('Test accuracy')
            plt.xlabel('Epoch')
            plt.savefig(plt_path)