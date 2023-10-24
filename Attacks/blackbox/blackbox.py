import sys
import wandb
import torch
import torchmetrics
from typing import Dict, Tuple
from rich.progress import Progress
from rich.pretty import pretty_repr
from Attacks.utils import generate_nohop_graph, init_shadow_loader, generate_attack_samples
from Attacks.blackbox.train_eval import train_sha, train_bbattack
from Attacks.dataset import Data
from Utils.console import console
from Utils.tracking import tracker_log
from Models.model import NN, CustomNN
from Models.utils import init_model

def attack(args, graphs:Tuple, tar_model:torch.nn.Module, device:torch.device, history:Dict, name:str):

    tar_g, sha_g = graphs

    with console.status("Initializing Shadow Loader") as status:
        with torch.no_grad():
            if args.num_class > 1:
                pred_fn = torch.nn.Softmax(dim=1).to(device)
            else:
                pred_fn = torch.nn.Sigmoid().to(device)

            shanh_g = generate_nohop_graph(graph=sha_g)
            sha_g = sha_g.to(device)
            shanh_g = shanh_g.to(device)
            tar_model.to(device)
            pred = tar_model.full(sha_g, sha_g.ndata['feat'])
            pred_nh = tar_model.full(shanh_g, shanh_g.ndata['feat'])
            sha_g.ndata['pred'] = pred_fn(pred)
            shanh_g.ndata['pred'] = pred_fn(pred_nh)
            console.log(f"Generated prediction on shadow graphs: {sha_g.ndata['pred'].size()}, and zero-hop shadow graph: {shanh_g.ndata['pred'].size()}")
            shatr_loader, shate_loader = init_shadow_loader(args=args, device=device, graph=sha_g)
            shanhtr_loader, shate_loader = init_shadow_loader(args=args, device=device, graph=shanh_g)

        console.log(f'Done Initializing Shadow Loader with size {len(shatr_loader)}, {len(shate_loader)}: :white_check_mark:')


    # init shadow model
    sha_model = init_model(args=args)
    shanh_model = init_model(args=args)

    # train shadow model
    sha_model = train_sha(args=args, loader=shatr_loader, model=sha_model, device=device, history=history, name=f'{name}_sha')
    shanh_model = train_sha(args=args, loader=shanhtr_loader, model=shanh_model, device=device, history=history, name=f'{name}_shanh')
    
    with console.status("Initializing Attack Data") as status:
        sha_model.to(device)
        shanh_model.to(device)

        with torch.no_grad():
            sha_pred = sha_model.full(sha_g, sha_g.ndata['feat'])
            shanh_pred = shanh_model.full(shanh_g, shanh_g.ndata['feat'])
            sha_g.ndata['sha_pred'] = sha_pred
            sha_g.ndata['shanh_pred'] = shanh_pred
            tarnh_g = generate_nohop_graph(graph=tar_g)
            tar_g = tar_g.to(device)
            tarnh_g = tarnh_g.to(device)
            pred = tar_model.full(tar_g, tar_g.ndata['feat'])
            pred_nh = tar_model.full(tarnh_g, tarnh_g.ndata['feat'])
            tar_g.ndata['pred'] = pred
            tar_g.ndata['nh_pred'] = pred_nh
            x_tr, y_tr, org_id_tr = generate_attack_samples(graph=sha_g, mode='shadow', device=device)
            x_te, y_te, org_id_te = generate_attack_samples(graph=tar_g, mode='target', device=device)
            new_dim = int(x_tr.size(dim=1) / 2)
        tr_data = Data(X=x_tr, y=y_tr, id=org_id_tr)
        te_data = Data(X=x_te, y=y_te, id=org_id_te)
        console.log(f'Done Initializing Attack Data: :white_check_mark:')

    atr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.att_bs, drop_last=True, shuffle=True)
    ate_loader = torch.utils.data.DataLoader(te_data, batch_size=args.att_bs, drop_last=False, shuffle=False)
    att_model = CustomNN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
    att_model = train_bbattack(args=args, tr_loader=atr_loader, te_loader=ate_loader, model=att_model, 
                                device=device, history=history, name= f'{name}_attack')
    
        # rprint(f"Attack model: {att_model}")

    with Progress(console=console) as progress:
        
        task1 = progress.add_task("[greed] Taking prediction...", total=len(ate_loader))
        threshold = [0.1*i for i in range(1,10)]
        metric = ['auc', 'acc', 'pre', 'rec', 'f1']
        objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        pred_fn = torch.nn.Sigmoid().to(device)
        node_dict = {}

        with torch.no_grad():
            label = torch.Tensor([]).to(device)
            preds = torch.Tensor([]).to(device)
            org_id = torch.Tensor([]).to(device)

            for bi, d in enumerate(ate_loader):
                features, target, idx = d
                features = features.to(device)
                target = target.to(device)
                predictions =  torch.squeeze(pred_fn(att_model(features)), dim=-1)
                label = torch.cat((label, target), dim=0)
                preds = torch.cat((preds, predictions), dim=0)
                org_id = torch.cat((org_id, idx), dim=0)
                progress.advance(task1)

        task2 = progress.add_task("[red] Assess with different threshold...", total=len(threshold))

        for thres in threshold:

            metric_dict = {
                'auc': torchmetrics.classification.BinaryAUROC().to(device),
                'acc': torchmetrics.classification.BinaryAccuracy(threshold=thres).to(device),
                'pre': torchmetrics.classification.BinaryPrecision(threshold=thres).to(device),
                'rec': torchmetrics.classification.BinaryRecall(threshold=thres).to(device),
                'f1': torchmetrics.classification.BinaryF1Score(threshold=thres).to(device)
            }

            results = {}
            for m in metric:
                met = metric_dict[m]
                perf = met(preds, label)
                results[f"Attack - best test/{m}"] = perf
                wandb.summary[f'Threshold: {thres}, BEST TEST {m}'] = perf
            tracker_log(dct=results)

            org_id = org_id.detach()
            preds = preds.detach()
            label = label.detach()

            for i, key in enumerate(org_id):
                if key.int().item() in node_dict.keys():
                    node_dict[key.item().item()]['pred'].append(int(preds[i].item() > thres))
                else:
                    node_dict[key] = {
                        'label': label[i].item(),
                        'pred': [int(preds[i].item() > thres)]
                    }
            progress.advance(task2)

        console.log(pretty_repr(node_dict))

        res_node_dict = {}
        for key in node_dict.keys():
            lab = node_dict[key]['label']
            t = 0
            for i in node_dict[key]['pred']:
                if i == lab: t+=1
            res_node_dict[f'{key}'] = f'{t}'
        wandb.summary[f'Node Correct / times'] = res_node_dict
        console.log(f"Done Evaluating best model: :white_check_mark:")
    return att_model, history