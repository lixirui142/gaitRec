from net.st_gcn import Model
import torch
from torch.utils.data import DataLoader
from data import TestDataset
import argparse

def test(model, epoch, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 0):
            torch.cuda.empty_cache()

            data, label = batch[0].squeeze(0).to(device), batch[1]
            pred = model(data)
            pred = pred[1:].argmax()
            if (pred == 0):
                correct += 1
            total += 1

            # correct += (torch.abs(pred - label) < 0.5).sum().cpu().numpy()
            # total += label.shape[1]
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%5d] accuracy: %.5f' %
                      (i + 1, correct / total))
    accuracy = correct / total
    print("Epoch %d: Total test accuracy: %.5f" % (epoch + 1, accuracy))
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph_args = {"layout": 'openpose', "strategy": 'spatial', "max_hop": 1, "dilation": 1}
    model = Model(3, graph_args, True)
    model = model.to(device)

    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    args = parser.parse_args()

    args.data_dir="G:/program/github/gait-gcn/data/prime-joints/"
    # args.data_dir="G:/xuke/CASIA-B/prime-joints/"
    # args.data_dir = "C:/Users/admn/Desktop/gait prp/gait-gcn/data/prime-joints/"
    args.threads = 1
    args.batchSize = 1
    args.lr = 0.001
    args.epoch = 150
    args.rampdown_epoch = 180
    args.save_dir = "model/"
    args.viewset = ["180"]
    args.counter_num = 8
    args.pos_num = 8
    args.clip_len = 70
    args.sample_list = [i for i in range(10)]
    args.id_list = [i for i in range(60)]
    args.test_list = [i for i in range(100, 124)]
    args.load_pretrain = True
    args.pretrain = "model/ckpt_149.pth"

    if args.load_pretrain:
        ckpt = torch.load(args.pretrain, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['state_dict'], strict=True)

    test_set = TestDataset(args)
    test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
    test(model,0,test_loader)
