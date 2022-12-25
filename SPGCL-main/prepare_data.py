from dtaidistance import dtw
import argparse
import os
import torch
import numpy as np
import lib.utils as utils
from lib.args import add_args
from lib.dataloader import Slope, Traffic


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


def prepare_data(data, save_dir=r"..\SPGCL\datasets"):
    save_path = os.path.join(save_dir, "{}_dtw.npy".format(data))
    if os.path.exists(save_path):
        return
    if "PEMS" in data:      # Traffic datasets dtw matrix needs long time to build, so we cut it in pieces
        slope_set = Traffic(save_dir, data_name=data, seq_len=12, pre_len=12, args=args, prepare_data=True)
        save_path = os.path.join(save_dir, data)
        for i in range(args.data_piece):
            save_path_i = os.path.join(save_path, "{}_dtw_{}.npy".format(data, i))
            if os.path.exists(save_path_i):
                pass
            else:
                dtw_list = dtw.distance_matrix(slope_set.features[:, i:i + args.seq_len], parallel=False, show_progress=True).astype(
                    "float32")
                np.save(save_path_i, dtw_list)
    else:   # Slope datasets
        slope_set = Slope(save_dir, data_name=data, seq_len=3, pre_len=1, args=args, prepare_data=True)
        features = slope_set.features
        dtw_list = dtw.distance_matrix(features, parallel=False, show_progress=True)
        np.save(save_path, dtw_list)


if __name__ == '__main__':
    # Do not forget to change args.data in lib.args.py
    save_dir = r"..\SPGCL\datasets"
    data = ["HZY_west", "HZY_east", "PEMS03", "PEMS04"]
    for i in data:
        prepare_data(i, save_dir)
