import argparse


def add_general_group(group):
    group.add_argument("--proj_name", type=str, default='', help="Name of the experiments", required=True)
    group.add_argument("--model_path", type=str, default="results/models/", help="directory path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="directory path for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--gen_mode", type=str, default='clean', help="Mode of running ['clean', 'dp']", required=True)
    group.add_argument("--gen_submode", type=str, default='ind', help="Mode of GNN ['trans', 'ind']", required=True)
    group.add_argument("--device", type=str, default='cpu', help="")
    group.add_argument("--debug", type=int, default=0)
    group.add_argument("--met", type=str, default='acc', help="Metrics of performance")

def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/datasets/', help="directory path to dataset")
    group.add_argument("--data_mode", type=str, default='none', help="Mode of preprocessing data", required=True)
    group.add_argument('--data', type=str, default='cora', help="name of dataset")
    group.add_argument("--dens", type=float, default=1.0, help="dropping rate")

def add_model_group(group):
    group.add_argument("--model", type=str, default='sage', help="Model type")
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--bs', type=int, default=512, help="batch size for training process")
    group.add_argument('--nnei', type=int, default=4, help="# of neighbor each layer")
    group.add_argument('--nlay', type=int, default=2, help='# of layers')
    group.add_argument('--hdim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--opt", type=str, default='adam')
    group.add_argument("--dout", type=float, default=0.2)
    group.add_argument("--pat", type=int, default=20)
    group.add_argument("--nhead", type=int, default=8)
    group.add_argument("--aggtype", type=str, default='gcn')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--retrain", type=int, default=0)

def add_dp_group(group):
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument("--clip_node", type=int, default=4, help='number of allowed appearance')
    group.add_argument("--trim_rule", type=str, default='adhoc', help='trimming rule')
    group.add_argument('--sampling_rate', type=float, default=0.08, help="batch size for training process")

def add_model_attack_group(group):
    group.add_argument("--att_mode", type=str, default='blackbox', help="Attack mode", required=True)
    group.add_argument("--att_submode", type=str, default='joint', help="Joint in shadow graph or not", required=False)
    group.add_argument('--att_lay', type=int, default=4, help='# of layers')
    group.add_argument('--att_hdim', type=int, default=64, help='hidden embedding dim')
    group.add_argument("--att_lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--att_bs', type=int, default=512, help="batch size for training process")
    group.add_argument("--att_epochs", type=int, default=100, help='training step')
    group.add_argument("--sha_lr", type=float, default=0.001, help="learning rate")
    group.add_argument("--sha_epochs", type=int, default=100, help='training step')
    group.add_argument("--sha_rat", type=float, default=0.5, help="learning rate")

def parse_args():
    parser = argparse.ArgumentParser()
    exp_grp = parser.add_argument_group(title="Attack setting")
    add_general_group(exp_grp)
    add_data_group(exp_grp)
    add_model_group(exp_grp)
    add_dp_group(exp_grp)
    add_model_attack_group(exp_grp)
    return parser.parse_args()
