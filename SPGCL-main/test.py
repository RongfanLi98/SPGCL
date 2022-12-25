# matplotlib.use('Agg')
import argparse
import os
import torch
import numpy as np
import time
import torch.optim as optim
import lib.utils as utils
from lib.args import add_args
from torch.utils.data import DataLoader, Subset
from lib.dataloader import Slope, Traffic
from lib.layers.SPGCL import SPGCL
from Trainers import SPGCLTrainer

# Load and initialize other parameters
parser = argparse.ArgumentParser('StdModel')
add_args(parser)
args = parser.parse_args()
args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.save_dir = os.path.join('results', args.data, args.log_key)
args.fig_save_dir = os.path.join(args.save_dir, 'figs')
args.log_dir = os.path.join(args.save_dir, 'logs')
args.graph_dir = os.path.join(args.save_dir, 'graphs')
utils.makedirs(args.save_dir)
utils.makedirs(args.fig_save_dir)
utils.makedirs(args.graph_dir)
utils.set_random_seed(args.seed)

if __name__ == '__main__':
    # initialize
    model = SPGCL(args=args).to(args.device)

    loss = torch.nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.mode = "test"
    args.feature_volume = 30  # test feature_volume

    normed_judge_list = []
    real_judge_list = []
    pems_result_list = []
    if "PEMS" in args.data:
        test_set = Traffic(data_dir=args.data_dir, data_name=args.data, seq_len=args.ini_seq_len, pre_len=args.pre_len,
                           args=args, train_set=False)
    else:
        test_set = Slope(data_dir=args.data_dir, data_name=args.data, seq_len=args.ini_seq_len, pre_len=args.pre_len,
                         args=args, train_set=False)

    scaler = test_set.scaler
    args.num_nodes = test_set.num_nodes

    test_set = Subset(test_set, range(0, test_set.len))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    train_loader = test_loader
    valid_loader = None

    trainer = SPGCLTrainer(model, loss, optimizer, train_loader, valid_loader, test_loader, scaler, args)

    if args.mode == "test":
        normed_judge, real_judge, pems_result = trainer.pure_test(model, trainer.args, test_loader, scaler,
                                                                  path=args.save_dir + r"\best_model.pth",
                                                                  mode=args.mode)
    else:
        raise ValueError

    normed_judge_list.append(normed_judge)
    real_judge_list.append(real_judge)
    pems_result_list.append(pems_result)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    normed_judge_list = np.array(normed_judge_list)
    real_judge_list = np.array(real_judge_list)
    pems_result_list = np.array(pems_result_list)
    normed_judge, real_judge, pems_result = normed_judge_list.mean(axis=0), real_judge_list.mean(
        axis=0), pems_result_list.mean(axis=0)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            normed_judge[0], normed_judge[1], normed_judge[2], normed_judge[3], normed_judge[4], time_stamp,
            "normed_value", args.data, args.log_key, args.mode)
        fin.write(result)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            real_judge[0], real_judge[1], real_judge[2], real_judge[3], real_judge[4], time_stamp, "real_value",
            args.data, args.log_key, args.mode)
        fin.write(result)
    with open(args.save_file, mode='a') as fin:
        result = "\n{:.3f},{:.3f},{:.3f},{:.3f},{},{},{},{},{}".format(
            pems_result[0], pems_result[1], pems_result[2], pems_result[3], time_stamp, "real_value",
            args.data, args.log_key, args.mode)
        fin.write(result)
