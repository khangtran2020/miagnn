import torch
import torchmetrics
from rich.progress import Progress
from typing import Dict
from Models.utils import EarlyStopping
from Utils.console import console
from Utils.tracking import tracker_log

def train_sha(args, loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict, name=str):
    
    mode = 'Shadow'
    if 'shanh' in name:
        mode = 'Shadow Zero Hop'

    model_name = '{}.pt'.format(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.num_class > 1:
        objective = torch.nn.CrossEntropyLoss().to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        objective = torch.nn.BCEWithLogitsLoss().to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = torchmetrics.classification.BinaryAccuracy().to(device)

    console.log(f"[green]{mode} / Optimizing model with optimizer[/green]: {optimizer}")
    console.log(f"[green]{mode} / Objective of the training process[/green]: {objective}")
    console.log(f"[green]{mode} / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]{mode} / Evaluating with metrics[/green]: {metrics}")

    model.to(device)
    model.train()
    es = EarlyStopping(patience=15, mode='max', verbose=False)

    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Training...", total=args.sha_epochs)
        task2 = progress.add_task("[green]Updating...", total=len(loader))

        # progress.reset(task_id=task1)
        for epoch in range(args.sha_epochs):
            tr_loss = 0
            ntr = 0
            # train
            for bi, d in enumerate(loader):
                model.zero_grad()
                _, _, mfgs = d
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["pred"]
                preds = model(mfgs, inputs)
                loss = objective(preds, labels)
                preds = pred_fn(preds)
                metrics.update(preds, labels.argmax(dim=1))
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()*preds.size(dim=0)
                ntr += preds.size(dim=0)
                progress.advance(task2)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task2)

            results = {
                f"{mode} / loss": tr_loss, 
                f"{mode} / acc": tr_perf,
            }

            if mode == 'Shadow Zero Hop':
                history[f'shanhtr_loss'].append(tr_loss)
                history[f'shanhtr_perf'].append(tr_perf)
            else:
                history[f'shtr_loss'].append(tr_loss)
                history[f'shtr_perf'].append(tr_perf)
            es(epoch=epoch, epoch_score=tr_perf, model=model, model_path=args.model_path + model_name)
            tracker_log(dct = results)
            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]acc[/yellow]: {tr_perf}") 
            progress.advance(task1)
        console.log(f"Done Training {mode}: :white_check_mark:")
    model.load_state_dict(torch.load(args.model_path + model_name))
    return model

def train_bbattack(args, tr_loader:torch.utils.data.DataLoader, te_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict, name=str):
    
    model_name = '{}.pt'.format(name)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.att_lr)
    # DEfining criterion
    objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
    objective.to(device)
    pred_fn = torch.nn.Softmax(dim=1).to(device)
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
            # train
            for bi, d in enumerate(tr_loader):
                model.zero_grad()
                features, target, _ = d
                features = features.to(device)
                target = target.to(device)
                predictions = model(features)
                loss = objective(predictions, target.float())
                loss.backward()
                opt.step()
                predictions = torch.squeeze(pred_fn(predictions), dim=-1)
                metrics.update(predictions, target)
                tr_loss += loss.item()*predictions.size(dim=0)
                ntr += predictions.size(dim=0)
                progress.advance(task2)
            
            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task2)

            te_loss = 0
            nte = 0

            # validation
            with torch.no_grad():

                for bi, d in enumerate(te_loader):
                    features, target, _ = d
                    features = features.to(device)
                    target = target.to(device)
                    predictions = model(features)
                    loss = objective(predictions, target.float())
                    predictions = torch.squeeze(pred_fn(predictions), dim=-1)
                    metrics.update(predictions, target)
                    te_loss += loss.item()*predictions.size(dim=0)
                    nte += predictions.size(dim=0)
                    progress.advance(task3)

            te_loss = te_loss / nte 
            te_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task3)

            results = {
                "Target train/loss": tr_loss, 
                f"Target train/auc": tr_perf, 
                "Target test/loss": te_loss, 
                f"Target test/auc": te_perf,
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
