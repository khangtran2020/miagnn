import torch
import torchmetrics
from rich.progress import Progress
from typing import Dict
from Models.utils import EarlyStopping
from Utils.console import console
from Utils.tracking import tracker_log

def train(args, tr_loader:torch.utils.data.DataLoader, te_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict, name=str):

    model_name = '{}.pt'.format(name)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.att_lr)
    # DEfining criterion
    objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
    objective.to(device)
    pred_fn = torch.nn.Sigmoid().to(device)
    metrics = torchmetrics.classification.BinaryAUROC().to(device)

    console.log(f"[green]Attack / Optimizing model with optimizer[/green]: {opt}")
    console.log(f"[green]Attack / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Attack / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Attack / Evaluating with metrics[/green]: {metrics}")

    model.to(device)
    model.train()
    es = EarlyStopping(patience=15, mode='max', verbose=False)

    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Training...", total=args.att_epochs)
        task2 = progress.add_task("[green]Updating...", total=len(tr_loader))
        task3 = progress.add_task("[green]Evaluating...", total=len(te_loader))

        for epoch in range(args.att_epochs):

            tr_loss = 0
            ntr = 0
            for bi, d in enumerate(tr_loader):
                model.zero_grad()
                features, target = d
                _, label, loss_tensor, out_dict, grad_dict = features
                feat = (label, loss_tensor, out_dict, grad_dict)
                target = torch.unsqueeze(target, dim=1).to(device)
                predictions = model(feat)
                loss = objective(predictions, target.float())
                loss.backward()
                opt.step()
                metrics.update(pred_fn(predictions), target)
                ntr += predictions.size(dim=0)
                tr_loss += loss.item()*predictions.size(dim=0)
                progress.advance(task2)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task2)

            te_loss = 0
            nte = 0

            model.eval()
            for bi, d in enumerate(te_loader):
                model.zero_grad()
                features, target = d
                _, label, loss_tensor, out_dict, grad_dict = features
                feat = (label, loss_tensor, out_dict, grad_dict)
                target = target.to(device)
                target = torch.unsqueeze(target, dim=1).to(device)
                predictions = model(feat)
                loss = objective(predictions, target.float())
                predictions = pred_fn(predictions)
                metrics.update(predictions, target)
                nte += predictions.size(dim=0)
                te_loss += loss.item()*predictions.size(dim=0)
                progress.advance(task3)

            te_loss = te_loss / nte
            te_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task3)

            results = {
                "Atack train/loss": tr_loss, 
                f"Atack train/auc": tr_perf, 
                "Atack test/loss": te_loss, 
                f"Atack test/auc": te_perf,
            }
            history['attr_loss'].append(tr_loss)
            history['attr_perf'].append(tr_perf)
            history['atte_loss'].append(te_loss)
            history['atte_perf'].append(te_perf)
            es(epoch=epoch, epoch_score=te_perf, model=model, model_path=args.model_path + model_name)
            tracker_log(dct=results)
            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]auc[/yellow]: {tr_perf}, [yellow]te_loss[/yellow]: {te_loss}, [yellow]te_auc[/yellow]: {te_perf}") 
            progress.advance(task1)
        console.log(f"Done Training Attack model: :white_check_mark:")
    model.load_state_dict(torch.load(args.model_path + model_name))
    return model
            
        
