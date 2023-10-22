import os
import torch
import time
import pickle
import random
import numpy as np
from contextlib import contextmanager
from Utils.console import console, log_table

@contextmanager
def timeit(logger, task):
    logger.info(f'Started task {task} ...')
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info(f'Completed task {task} - {(t1 - t0):.3f} sec.')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_dict(path, dct):
    with open(path, 'wb') as f:
        pickle.dump(dct, f)

def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]

def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]

def get_index_by_list(arr, test_arr):
    return torch.isin(arr, test_arr).nonzero(as_tuple=True)[0]

def get_index_by_not_list(arr, test_arr):
    return (1 - torch.isin(arr, test_arr).int()).nonzero(as_tuple=True)[0]

def print_args(args):
    arg_dict = {}
    keys = ['gen_mode', 'gen_submode', 'seed', 'met', 'data', 'data_mode', 'dens', 'bs',
            'nnei', 'model', 'lr', 'nlay', 'epochs', 'clip', 'clip_node', 'trim_rule', 'ns', 'debug', 'device',
            'att_mode', 'att_submode', 'att_lay', 'att_hdim', 'att_lr', 'att_bs', 'att_epochs', 'sha_lr', 'sha_epochs', 'sha_rat']
    if args.data_mode == 'none':
        keys.remove('dens')

    if args.general_mode == 'clean':
        keys.remove('clip')
        keys.remove('clip_node')
        keys.remove('trim_rule')
        keys.remove('ns')

    for key in keys:
        arg_dict[key] = str(getattr(args, key))
    log_table(dct=arg_dict, name='Arguments')
    return arg_dict

def read_pickel(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def init_history(args):

    data_hist = {
        'tr_id': None,
        'va_id': None,
        'te_id': None,
    }

    if args.gen_mode == 'clean':
        target_model_hist = {
            'name': None,
            'train_history_loss': [],
            'train_history_acc': [],
            'val_history_loss': [],
            'val_history_acc': [],
            'test_history_loss': [],
            'test_history_acc': [],
            'best_test': 0
        }
    else:
        target_model_hist =  {
            'name': None,
            'train_history_loss': [],
            'train_history_acc': [],
            'val_history_loss': [],
            'val_history_acc': [],
            'test_history_loss': [],
            'test_history_acc': [],
            '% subgraph': [],
            '% node avg': [],
            '% edge avg': [],
            'avg rank': []
        }

    att_hist = {
        'str_mask': None,
        'ste_mask': None,
        'shtr_loss': [],
        'shtr_perf': [],
        'shanhtr_loss': [],
        'shanhtr_perf': [],
        'attr_loss': [],
        'attr_perf': [],
        'atva_loss': [],
        'atva_perf': [],
        'atte_loss': [],
        'atte_perf': [],
    }

    return data_hist, target_model_hist, att_hist

def get_name(args, current_date):

    date_str = f'{current_date.day}{current_date.month}{current_date.year}-{current_date.hour}{current_date.minute}'
    data_keys = ['data', 'seed', 'data_mode', 'dens']
    model_keys = ['data', 'gen_mode', 'seed', 'nnei', 'model', 'lr', 'nlay', 'hdim', 'epochs', 
                  'opt', 'clip', 'clip_node', 'trim_rule', 'ns', 'sampling_rate']
    gen_keys = ['data', 'gen_mode', 'data_mode', 'dens', 'seed', 'nnei', 
                'model', 'nlay', 'clip', 'clip_node', 'trim_rule', 'ns', 
                'sampling_rate', 'att_mode', 'sha_rat']
    att_keys = ['att_mode', 'att_submode', 'seed', 'data', 'gen_mode', 'gen_submode', 'data', 'dens', 'sha_rat',
                'model', 'nlay', 'sampling_rate', 'clip', 'clip_node', 'trim_rule', 'ns']
        
    if args.data_mode != 'density': 
        gen_keys.remove('dens')
        data_keys.remove('dens')
        att_keys.remove('dens')
    if args.general_mode == 'clean':
        
        gen_keys.remove('clip')
        gen_keys.remove('clip_node')
        gen_keys.remove('trim_rule')
        gen_keys.remove('ns')
        gen_keys.remove('sampling_rate')

        model_keys.remove('clip')
        model_keys.remove('clip_node')
        model_keys.remove('trim_rule')
        model_keys.remove('ns')
        model_keys.remove('sampling_rate')
        
        att_keys.remove('clip')
        att_keys.remove('clip_node')
        att_keys.remove('trim_rule')
        att_keys.remove('ns')
        att_keys.remove('sampling_rate')

    general_str = ''
    for key in gen_keys:
        general_str += f"{key}_{getattr(args, key)}_"
    general_str += date_str

    data_str = ''
    for key in data_keys:
        data_str += f"{key}_{getattr(args, key)}_"
    
    model_str = ''
    for key in model_keys:
        model_str += f"{key}_{getattr(args, key)}_"

    att_str = ''
    for key in att_keys:
        att_str += f"{key}_{getattr(args, key)}_"

    name = {
        'data': data_str[:-1],
        'model': model_str[:-1],
        'att': att_str[:-1],
        'general': general_str
    }

    return name
