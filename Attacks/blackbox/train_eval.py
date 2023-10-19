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
                tr_loss += loss.item()
                ntr += preds.size(dim=0)
                progress.advance(task2)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task2)

            results = {
                f"{mode} / loss": tr_loss, 
                f"{mode} /acc": tr_perf,
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
