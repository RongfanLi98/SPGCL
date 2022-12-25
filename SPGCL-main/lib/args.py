import os
import configparser

# datasets path
DATAPATH = {
    "HZY_west": r"..\SPGCL\datasets\HZY_west.csv",
    "HZY_east": r"..\SPGCL\datasets\HZY_east.csv",
    "PEMS03":   r"..\SPGCL\datasets",
    "PEMS04":   r"..\SPGCL\datasets",
}


def add_args(parser, data_set="PEMS03"):
    # Data set settings.  HZY_west, HZY_east, PEMS03, PEMS04
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--log_key', type=str, default="training", help="The log dir")
    parser.add_argument('--conti', type=eval, default=True, help="Whether to continue training")

    parser.add_argument('--debug', type=eval, default=False, help="Whether to debug, using simple training")
    parser.add_argument('--valid', type=eval, default=False, help="Whether to use validation")
    parser.add_argument('--update_freq', type=int, default=10, help="frequency of update graph")

    config_path = r'..\SPGCL\CONFIG\{}.conf'.format(data_set)
    if os.path.exists(config_path):
        load_paras_config(parser, config_path)
    else:
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--seed', type=int, default=12343, help='Seed for initializing training. ')
        parser.add_argument('--epochs', type=int, default=2000)
        parser.add_argument('--early_stop_patience', type=int, default=20, help='Patient for early stop.')
        parser.add_argument('--early_stop_patience_K', type=int, default=50, help='Patient for early stop.')

        parser.add_argument('--log_freq', type=int, default=1, help="frequency of predict_loss log.")
        parser.add_argument('--val_freq', type=int, default=5, help="frequency of validating and saving")

        parser.add_argument('--sparse', type=eval, default=True, help="sparse adj matrix")
        parser.add_argument('--positive_per_K', default=2, type=int, help='The number of edges be absorbed in every k')

        parser.add_argument('--density', default=0.5, type=float, help='the density of graph')
        parser.add_argument('--delta', default=0.6, type=float, help='the threshold of positive edges')
        parser.add_argument('--delta_negative', default=0.3, type=float, help='the threshold of negative edges')

        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--test_batch_size', type=int, default=1)
        parser.add_argument('--data_volume', default=2000, type=int)
        parser.add_argument('--feature_volume', default=18, type=int)

        parser.add_argument('--gamma_1', default=1, type=float)
        parser.add_argument('--gamma_2', default=1, type=float)
        parser.add_argument('--eta', default=0.3, type=float)
        parser.add_argument('--Q', default=0.3, type=float, help="[0,0.5]")
        parser.add_argument('--nu_samples', default=500, type=float)

        parser.add_argument('--train_loop', type=int, default=20)
        parser.add_argument('--data_piece', type=int, default=18, help="data time length")

        parser.add_argument('--acc_threshold', type=float, default=0.1)
        parser.add_argument('--acc_real_threshold', type=float, default=5)

        parser.add_argument('--pre_len', type=int, default=12, help="len of Y")
        parser.add_argument('--ini_seq_len', type=int, default=12, help="ini len of S")
        parser.add_argument('--seq_len', type=int, default=13, help="len of S + 3")

    parser.add_argument('--stage_mean', default=1., type=float, help='mean score of a new k')
    parser.add_argument("-f", type=str, default="")
    parser.add_argument("--f", type=str, default="")
    parser.add_argument("--ip", type=str, default="")
    parser.add_argument("--stdin", type=str, default="")
    parser.add_argument("--control", type=str, default="")
    parser.add_argument("--hb", type=str, default="")
    parser.add_argument("--Session.signature_scheme", type=str, default="")
    parser.add_argument("--Session.key", type=str, default="")
    parser.add_argument("--shell", type=str, default="")
    parser.add_argument("--transport", type=str, default="")
    parser.add_argument("--iopub", type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grad_norm', type=eval, default=False, help="whether to use clip_grad_norm_")
    # Model parameters
    parser.add_argument('--positive_init', default=10, type=int, help='init positive number')
    parser.add_argument('--negative_init', default=10, type=int, help='init negative number')

    parser.add_argument('--real_value', type=eval, default=False, help="predict real label or after normalizing")
    parser.add_argument('--min_update_nodes', default=5, type=int, help='if lr is too low, early stop')
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--data', choices=DATAPATH.keys(), type=str, default=data_set)
    parser.add_argument('--data_path', choices=DATAPATH, type=str, default=DATAPATH[data_set])
    parser.add_argument('--data_dir', type=str, default=r"..\SPGCL\datasets")

    parser.add_argument('--train', type=float, default=0.5)
    parser.add_argument('--validation', type=float, default=0.3)
    parser.add_argument('--test', type=float, default=0.2)

    # Saving settings
    parser.add_argument('--save_file', type=str, default=r"..\SPGCL\results\pure_result.csv")
    parser.add_argument('--writer', type=eval, default=True, help="tensorboard")
    # no update
    parser.add_argument('--lambda_n', type=int, default=10)
    parser.add_argument('--lambda_p', type=int, default=10)


def load_paras_config(parser, config_path):
    # -----------------------------------------------------------------------------#
    # get configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    print("++++++++++++++")
    print(config_path)  # using PEMS_{}.conf as "config"!
    print("++++++++++++++")
    # -----------------------------------------------------------------------------#
    # hyper
    parser.add_argument('--lr', default=config['hyper']['lr'], type=float)
    parser.add_argument('--weight_decay', default=config['hyper']['weight_decay'], type=float)
    parser.add_argument('--seed', default=config['hyper']['seed'], type=int)
    parser.add_argument('--epochs', default=config['hyper']['epochs'], type=int)
    parser.add_argument('--early_stop_patience', default=config['hyper']['early_stop_patience'], type=int)
    parser.add_argument('--early_stop_patience_K', default=config['hyper']['early_stop_patience_K'], type=int)
    # model
    parser.add_argument('--sparse', default=config['Model']['sparse'], type=eval)
    parser.add_argument('--positive_per_K', default=config['Model']['positive_per_K'], type=int)
    parser.add_argument('--positives', default=config['Model']['positives'], type=int)
    parser.add_argument('--density', default=config['Model']['density'], type=float)
    parser.add_argument('--delta', default=config['Model']['delta'], type=float, help="delta")
    parser.add_argument('--delta_negative', default=config['Model']['delta_negative'], type=float, help="delta negative")
    # data
    parser.add_argument('--batch_size', default=config['data']['batch_size'], type=int)
    parser.add_argument('--test_batch_size', default=config['data']['test_batch_size'], type=int)
    parser.add_argument('--train_loop', default=config['data']['train_loop'], type=int)
    parser.add_argument('--data_piece', default=config['data']['data_piece'], type=int)
    parser.add_argument('--data_volume', default=config['data']['data_volume'], type=int, help="subgraph node number")
    parser.add_argument('--feature_volume', default=config['data']['feature_volume'], type=int)
    parser.add_argument('--gamma_1', default=config['data']['gamma_1'], type=float, help="weight of loss1")
    parser.add_argument('--gamma_2', default=config['data']['gamma_2'], type=float, help="weight of loss2")
    parser.add_argument('--eta', default=config['data']['eta'], type=float, help="PU-Learning prior probablity")
    parser.add_argument('--Q', default=config['data']['Q'], type=float, help="Q*num pieces of unlabel points to choose")
    parser.add_argument('--nu_samples', default=config['data']['nu_samples'], type=int, help="sample num in pu learning")
    parser.add_argument('--pre_len', type=int, default=config['data']['pre_len'], help="len of Y")
    parser.add_argument('--ini_seq_len', type=int, default=config['data']['ini_seq_len'])
    parser.add_argument('--seq_len', type=int, default=config['data']['seq_len'], help="len of S + 3")
    # save
    parser.add_argument('--acc_threshold', default=config['save']['acc_threshold'], type=float)
    parser.add_argument('--acc_real_threshold', default=config['save']['acc_real_threshold'], type=float)
    # log
    parser.add_argument('--log_freq', default=config['log']['log_freq'], type=int)
    parser.add_argument('--val_freq', default=config['log']['val_freq'], type=int)

