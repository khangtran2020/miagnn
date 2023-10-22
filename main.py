import os
import dgl
import torch
import datetime
import warnings
from config import parse_args
from Attacks.blackbox.blackbox import attack as bb_attack
from Data.read import read_data, whitebox_split, blackbox_split
from Data.utils import init_loader
from Models.process import train, evaluate
from Models.utils import init_model
from Utils.utils import seed_everything, read_pickel, print_args, init_history, get_name, save_dict
from Utils.console import console, log_table
from Utils.tracking import init_tracker, tracker_log_table

warnings.filterwarnings("ignore")

def run(args, current_time, device):

    data_hist, model_hist, att_hist = init_history(args=args)
    name = get_name(args=args, current_date=current_time)
    exist_data = False
    exist_model = False

    # read data 
    with console.status("Initializing Data") as status:
        data_name = f"{name['data']}.pkl"
        data_path = args.res_path + data_name
        if (os.path.exists(data_path)) & (args.retrain == 0):
            data_hist = read_pickel(file=data_path)
            exist_data = True
            console.log(f"History exist: :white_check_mark:, exist_data set to: {exist_data}")
        else:
            console.log(f"History exist: :x:, exist_data set to: {exist_data}")

        tar_g, sha_g = read_data(args=args, history=data_hist, exist=exist_data)
        console.log(f"Done Reading data: :white_check_mark:")

    with console.status("Initializing Shadow Data") as status:
        if args.att_mode == 'blackbox':
            sha_g = blackbox_split(graph=sha_g, history=data_hist, exist=exist_data)
        elif args.att_mode == 'whitebox':
            shadow_graph = whitebox_split(graph=sha_g, ratio=args.sha_ratio, history=data_hist, exist=exist_data)
        console.log(f"Done Initializing Shadow Data: :white_check_mark:")

    with console.status("Initializing Target Model") as status:
        model_name = f"{name['model']}.pt"
        model_path = args.model_path + model_name
        target_model_name = f"{name['model']}.pkl"
        target_model_path = args.res_path + target_model_name
        exist_model = (os.path.exists(model_path)) & (args.retrain == 0) & (os.path.exists(target_model_path))

        if exist_model:
            console.log(f"Model existed: :white_check_mark:")
            target_model_name = f"{name['model']}.pkl"
            target_model_path = args.res_path + target_model_name
            model_hist = read_pickel(file=target_model_path)
        else:
            console.log(f"Model did not exist: :x:")

        model = init_model(args=args)
        if exist_model: 
            model.load_state_dict(torch.load(model_path))
            console.log(f"Model exist, loaded previous trained model")
        console.log(f"Target model's configuration: {model}")
        
    if exist_model == False:
        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, graph=tar_g)
        model, model_hist = train(args=args, tr_loader=tr_loader, va_loader=va_loader, model=model, device=device, history=model_hist, name=name['model'])
        evaluate(args=args, te_loader=te_loader, model=model, device=device, history=model_hist)

    if args.att_mode == 'blackbox':
        att_model, att_hist = bb_attack(args=args, graphs=(tar_g, sha_g), tar_model=model, device=device, history=att_hist, name=name['att'])
        
    general_hist = {
        'data': data_hist,
        'model': model_hist,
        'att': att_hist
    }
    general_path = args.res_path + f"{name['general']}.pkl"
    save_dict(path=general_path, dct=general_hist)
    console.log(f"Saved result at path {general_path}.")

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.proj_name}")
    with console.status("Initializing...") as status:
        console.log(f'[bold][green]Initializing')
        arg_dict = print_args(args=args)
        init_tracker(name=args.project_name, config=arg_dict)
        tracker_log_table(dct=arg_dict, name='config')
        seed_everything(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'cpu':
            device = torch.device('cpu')
        console.log(f"Device running: {device}")
        console.log(f'[bold][green]Done!')
    run(args=args, current_time=current_time, device=device)
