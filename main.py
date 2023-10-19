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
from Utils.utils import seed_everything, read_pickel, print_args, init_history, get_name
from Utils.console import console, log_table
from Utils.tracking import init_tracker, tracker_log_table

# from Attacks.Runs.black_box import run as blackbox
# from Attacks.Runs.white_box import run as whitebox
# from Attacks.Runs.wb_simple import run as wanal
# from Attacks.Utils.utils import print_args, init_history, get_name, save_dict
# from Attacks.Utils.data_utils import shadow_split, shadow_split_whitebox_extreme, shadow_split_whitebox, read_data, shadow_split_whitebox_subgraph, shadow_split_whitebox_drop, shadow_split_whitebox_drop_ratio
# from Models.init import init_model

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
        
        if args.general_submode == 'ind':
            train_g, val_g, test_g, graph = read_data(args=args, history=data_hist, exist=exist_data)
        else:
            graph = read_data(args=args, history=data_hist, exist=exist_data)
        console.log(f"Done Reading data: :white_check_mark:")

    with console.status("Initializing Shadow Data") as status:
        if args.att_mode == 'blackbox':
            shadow_graph = blackbox_split(graph=train_g, ratio=args.sha_ratio, history=data_hist, exist=exist_data)
        elif args.att_mode == 'whitebox':
            shadow_graph = whitebox_split(graph=graph, ratio=args.sha_ratio, history=data_hist, exist=exist_data, diag=True)
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
        if args.general_submode == 'ind':
            tr_loader, va_loader, te_loader = init_loader(args=args, device=device, graphs=(train_g, val_g, test_g))
        else:
            tr_loader, va_loader, te_loader = init_loader(args=args, device=device, graphs=graph)
        model = train(args=args, tr_loader=tr_loader, va_loader=va_loader, model=model, device=device, history=model_hist, name=name['model'])
        evaluate(args=args, te_loader=te_loader, model=model, device=device, history=model_hist)

    if args.att_mode == 'blackbox':
        if args.general_submode == 'ind':
            graphs = (train_g, val_g, test_g, shadow_graph)
        else:
            graphs = (graph, shadow_graph)
        bb_attack(args=args, graphs=graphs, tar_model=model, device=device, history=att_hist, name=name['att'])
    # elif args.att_mode == 'whitebox':
    #     model_hist, att_hist = whitebox(args=args, graph=(train_g, val_g, test_g, shadow_graph), model=model, device=device, history=history, name=name)
    # elif args.att_mode == 'wanal':
    #     model_hist, att_hist = wanal(args=args, graph=(train_g, val_g, test_g, shadow_graph), model=model, device=device, history=history, name=name)

    # """
    #     SAVE RUNNING HISTORY TO FILE
    # """

    # general_hist = {
    #     'data': data_hist,
    #     'model': model_hist,
    #     'att': att_hist
    # }
    # general_path = args.res_path + f"{name['general']}.pkl"
    # save_dict(path=general_path, dct=general_hist)
    # rprint(f"Saved result at path {general_path}")

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    console.rule(f"Begin experiment: {args.project_name}")
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
