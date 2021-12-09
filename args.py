import argparse
from rec import REC_Processor

def get_args(null_args = False):
    parser = REC_Processor.get_parser()
    if null_args:
        args = parser.parse_args(args = [])
    else:
        args = parser.parse_args()

    #args.data_dir = "G:\program\github\gaitRec\data\gait_prime_joints_process"
    # args.data_dir="G:/xuke/CASIA-B/prime-joints/"
    args.data_dir = "G:/lxr/gaitRec/data/prime_joints/"
    # args.data_dir="C:/Users/admn/Desktop/gait prp/gait-gcn/data/prime-joints/"
    ## 001-bg-01-000
    args.threads = 1
    args.batchSize = 64
    args.lr = 0.01
    args.epoch = 200
    args.gamma = 0.1
    args.decay_epoch = args.epoch
    args.name = "test_dvs"
    args.save_dir = "model/" + args.name
    args.result_dir = "result/" + args.name
    # args.viewset = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
    args.viewset = ["090"]
    args.viewlen = 9
    args.clip_len = 40
    # args.sample_list = [i for i in range(10 * len(args.viewset))]
    # args.probe_nm_sample = [8, 9]
    # args.probe_bg_sample = [0, 1]
    # args.probe_cl_sample = [2, 3]
    # args.gallary_sample = [4, 5, 6, 7]
    args.sample_list = [i for i in range(args.viewlen * len(args.viewset))]
    args.probe_bg_sample = [0, 1]
    args.probe_nm_sample = [6, 7]
    args.probe_cl_sample = [3, 4]
    args.gallary_sample = [3, 5, 8]
    args.id_list = [i for i in range(10)]
    args.train_len = len(args.id_list)
    args.test_list = [i for i in range(10, 13)]
    args.load_pretrain = False
    args.pretrain = "model/090rm2/ckpt_best_0.91398.pth"
    args.class_rate = 1.0
    args.evaluate = False
    args.batch_class_num = 8
    args.class_sample_num = 4
    args.save_freq = 10
    args.alpha = 0.0
    args.center_lr = 1.0
    args.center_startep = 25
    args.enable_center = False
    args.wandb = True
    args.only_nm = True

    return args