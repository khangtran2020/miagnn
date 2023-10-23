import sys
import wandb
import torch
import torchmetrics
from typing import Dict, Tuple
from functools import partial
from rich.progress import Progress
from Attacks.dataset import ShadowData, custom_collate
from Attacks.whitebox.train_eval import train
from Utils.console import console
from Utils.tracking import tracker_log
from Models.model import WbAttacker

def attack(args, graphs:Tuple, tar_model:torch.nn.Module, device:torch.device, history:Dict, name:str):

    tar_g, sha_g = graphs

    with console.status("Initializing Attack Loader") as status:
        
        shtr_dataset = ShadowData(graph=sha_g, model=tar_model, num_layer=args.nlay, device=device, mode='train', nnei=-1)
        shte_dataset = ShadowData(graph=sha_g, model=tar_model, num_layer=args.nlay, device=device, mode='test', nnei=-1)

        label, weight = shtr_dataset.label_weight
        lab_weight = 1 - weight / weight.sum()
        console.log(f"Label weight will be: {lab_weight}")

        out_keys = [f'out_{i}' for i in range(args.nlay)]
        out_dim = []
        if args.nlay > 1:
            out_dim.append(args.hdim)
            for i in range(0, args.nlay - 2):
                out_dim.append(args.hdim)
            out_dim.append(args.num_class)
        else:
            out_dim.append(args.num_class)

        model_keys = []
        grad_dim = []
        for named, p in tar_model.named_parameters():
            if p.requires_grad:
                model_keys.append(named.replace('.', '-'))
                if 'bias' in named:
                    out_d = list(p.size())[0]
                    grad_dim.append((1, out_d))
                else:
                    out_d, in_d = list(p.size())
                    grad_dim.append((in_d, out_d))
                console.log(f"Model parameter {named} has size: {p.size()}")
        
        collate_fn = partial(custom_collate, out_key=out_keys, model_key=model_keys, device=device, num_class=args.num_class)
        tr_loader = torch.utils.data.DataLoader(shtr_dataset, batch_size=args.att_bs, collate_fn=collate_fn,
                                                drop_last=True, shuffle=True)
        te_loader = torch.utils.data.DataLoader(shte_dataset, batch_size=args.att_bs, collate_fn=collate_fn,
                                                drop_last=False, shuffle=False)
        
        console.log(f"Out dim: {out_dim}")
        console.log(f"Grad dim: {grad_dim}")
        x, y = next(iter(tr_loader))
        _, label, loss, out_dict, grad_dict = x
        console.log(f"Label size: {label.size()}")
        console.log(f"Loss size: {loss.size()}")
        console.log(f"Membership Label size: {y.size()}")
        for key in out_keys:
            console.log(f"Out dict at key {key} has size: {out_dict[key].size()}")
        for key in model_keys:
            console.log(f"Grad dict at key {key} has size: {grad_dict[key].size()}")
        # sys.exit()
        console.log(f'Done Initializing Attack Loader: :white_check_mark:')

    att_model = WbAttacker(label_dim=args.num_class, loss_dim=1, out_dim_list=out_dim, grad_dim_list=grad_dim, 
                           out_keys=out_keys, model_keys=model_keys, num_filters=4, device=device)
    att_model = train(args=args, tr_loader=tr_loader, te_loader=te_loader, model=att_model, device=device, 
                      history=history, name=f'{name}_attack')
    with Progress(console=console) as progress:
        task1 = progress.add_task("[greed] Taking prediction...", total=len(te_loader))
        threshold = [0.1*i for i in range(1,10)]
        metric = ['auc', 'acc', 'pre', 'rec', 'f1']
        objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        pred_fn = torch.nn.Sigmoid().to(device)
        node_dict = {}

        label = torch.Tensor([]).to(device)
        preds = torch.Tensor([]).to(device)
        org_id = torch.Tensor([]).to(device)

        for bi, d in enumerate(te_loader):
            features, target = d
            idx, label, loss_tensor, out_dict, grad_dict = features
            feat = (label, loss_tensor, out_dict, grad_dict)
            target = target.to(device)
            predictions = att_model(feat)
            predictions = torch.nn.functional.sigmoid(predictions)
            predictions = torch.squeeze(predictions, dim=-1)

            org_id = torch.cat((org_id, idx.detach()), dim=0)
            preds = torch.cat((preds, predictions.detach()), dim=0)
            label = torch.cat((label, target.detach()), dim=0)

        task2 = progress.add_task("[red] Assess with different threshold...", total=9)
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
                if key.item() in node_dict.keys():
                    node_dict[key.item()]['pred'].append(int(preds[i].item() > thres))
                else:
                    node_dict[key] = {
                        'label': label[i].item(),
                        'pred': [int(preds[i].item() > thres)]
                    }
            progress.advance(task2)

        res_node_dict = {}
        for key in node_dict.keys():
            lab = node_dict[key]['label']
            t = 0
            for i in node_dict[key]['pred']:
                if i == lab: t+=1
            res_node_dict[f'{key}'] = f'{t}'
        wandb.summary[f'Node Correct / times'] = res_node_dict
        console.log(f"Done Evaluating best model: :white_check_mark:")