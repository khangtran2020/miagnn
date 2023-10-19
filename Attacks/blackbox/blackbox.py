import sys
import torch
import torchmetrics
from typing import Dict, Tuple
from rich import print as rprint
from Attacks.utils import generate_nohop_graph, init_shadow_loader
from Attacks.blackbox.train_eval import train_sha
from Utils.console import console
from Models.model import NN, CustomNN
from Models.utils import init_model

def attack(args, graphs:Tuple, tar_model:torch.nn.Module, device:torch.device, history:Dict, name:str):

    if args.general_submode == 'ind':
        tr_g, te_g, sha_g = graphs
    else:
        g, sha_g = graphs

    with console.status("Initializing Shadow Loader") as status:
        with torch.no_grad():

            if args.num_class > 1:
                pred_fn = torch.nn.Softmax(dim=1).to(device)
            else:
                pred_fn = torch.nn.Sigmoid().to(device)

            if args.att_submode == 'joint':
                shanh_g = generate_nohop_graph(graph=sha_g)
                sha_g = sha_g.to(device)
                shanh_g = shanh_g.to(device)
                tar_model.to(device)
                pred_sha = tar_model.full(sha_g, sha_g.ndata['feat'])
                pred_shanh = tar_model.full(shanh_g, shanh_g.ndata['feat'])
                sha_g.ndata['pred'] = pred_fn(pred_sha)
                shanh_g.ndata['pred'] = pred_fn(pred_shanh)
                console.log(f'Generated prediction on shadow graphs and zero-hop shadow graph')
                shatr_loader, shate_loader = init_shadow_loader(args=args, device=device, graph=sha_g)
                shanhtr_loader, shanhte_loader = init_shadow_loader(args=args, device=device, graph=shanh_g)
            else:
                pass
        console.log(f'Done Initializing Shadow Loader: :white_check_mark:')


    # init shadow model
    sha_model = init_model(args=args)
    shanh_model = init_model(args=args)

    # train shadow model
    sha_model = train_sha(args=args, loader=shatr_loader, model=sha_model, device=device, history=history, name=f'{name}_sha')
    shanh_model = train_sha(args=args, loader=shanhtr_loader, model=shanh_model, device=device, history=history, name=f'{name}_shanh')
    
    # sys.exit()
    # with timeit(logger=logger, task='preparing-attack-data'):
    
    #     shadow_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_hops_shadow.pt"))
    #     shadow_model_nohop.load_state_dict(torch.load(args.save_path + f"{name['att']}_nohop_shadow.pt"))
        
    #     with torch.no_grad():

    #         shadow_model.to(device)
    #         shadow_model_nohop.to(device)
    #         shadow_conf = shadow_model.full(train_g, train_g.ndata['feat'])
    #         shadow_conf_nohop = shadow_model_nohop.full(train_g_nohop, train_g_nohop.ndata['feat'])
    #         train_g.ndata['shadow_conf'] = shadow_conf

    #         x, y = generate_attack_samples(graph=train_g, conf=shadow_conf, nohop_conf=shadow_conf_nohop, mode='shadow', device=device)
    #         x_test, y_test = generate_attack_samples(graph=train_g, conf=tr_conf, nohop_conf=tr_conf_nohop, mode='target', 
    #                                                  te_graph=test_g, te_conf=te_conf, te_nohop_conf=te_conf_nohop, device=device)
            
    #         test_distribution_shift(x_tr=x, x_te=x_test)
    #         x = torch.cat([x, x_test], dim=0)
    #         y = torch.cat([y, y_test], dim=0)
    #         for i in range(x.size(dim=1)):
    #             x[:, i] = (x[:,i] - x[:,i].mean()) / (x[:,i].std() + 1e-12)
    #         num_test = x_test.size(0)
    #         num_train = int((x.size(0) - num_test) * 0.8)

    #         new_dim = int(x.size(dim=1)/2)
    #         # train test split

    #         tr_data = Data(X=x[:num_train], y=y[:num_train])
    #         va_data = Data(X=x[num_train:-num_test], y=y[num_train:-num_test])
    #         te_data = Data(X=x[-num_test:], y=y[-num_test:])

    # # device = torch.device('cpu')
    # with timeit(logger=logger, task='train-attack-model'):

    #     tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.att_batch_size,
    #                                             pin_memory=False, drop_last=True, shuffle=True)

    #     va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.att_batch_size, num_workers=0, shuffle=False,
    #                                             pin_memory=False, drop_last=False)

    #     te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.att_batch_size, num_workers=0, shuffle=False,
    #                                             pin_memory=False, drop_last=False)
        
    #     attack_model = CustomNN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
    #     attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.att_lr)

    #     attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
    #                                 attack_model=attack_model, epochs=args.att_epochs, optimizer=attack_optimizer,
    #                                 name=name['att'], device=device, history=att_hist)

    # attack_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_attack.pt"))

    # metric = ['auc', 'acc', 'pre', 'rec', 'f1']
    # metric_dict = {
    #     'auc': torchmetrics.classification.BinaryAUROC().to(device),
    #     'acc': torchmetrics.classification.BinaryAccuracy().to(device),
    #     'pre': torchmetrics.classification.BinaryPrecision().to(device),
    #     'rec': torchmetrics.classification.BinaryRecall().to(device),
    #     'f1': torchmetrics.classification.BinaryF1Score().to(device)
    # }
    # for met in metric:
    #     te_loss, te_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
    #                                    metrics=metric_dict[met], criterion=torch.nn.BCELoss())
    #     rprint(f"Attack {met}: {te_auc}")
    
    # return model_hist, att_hist
