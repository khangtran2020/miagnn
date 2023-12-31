import dgl
import torch
import torchmetrics
from rich.progress import Progress
from typing import Dict
from Models.utils import EarlyStopping, draw_conf, draw_loss_grad, draw_full
from Utils.console import console
from Utils.tracking import tracker_log, wandb

def train(args, tr_loader:torch.utils.data.DataLoader, va_loader:torch.utils.data.DataLoader, tar_g:dgl.DGLGraph, sha_g:dgl.DGLGraph,
          model:torch.nn.Module, device:torch.device, history:Dict, name:str, name_pos:str=None):
    
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

    console.log(f"[green]Train / Optimizing model with optimizer[/green]: {optimizer}")
    console.log(f"[green]Train / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Train / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Train / Evaluating with metrics[/green]: {metrics}")

    model.to(device)
    model.train()

    es = EarlyStopping(patience=15, mode='min', verbose=False)

    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Training...", total=args.epochs)
        task2 = progress.add_task("[green]Updating...", total=len(tr_loader))
        task3 = progress.add_task("[cyan]Evaluating...", total=len(va_loader))

        # progress.reset(task_id=task1)
        if args.debug == 1:
            image_conf = []
            image_grad = []
            image_loss = []

            image_conf_full = []
            image_grad_full = []
            image_loss_full = []

        indx = 0
        for epoch in range(args.epochs):
            tr_loss = 0
            ntr = 0
            num_step = len(tr_loader)

            # train
            for bi, d in enumerate(tr_loader):
                model.zero_grad()
                _, _, mfgs = d
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                preds = model(mfgs, inputs)
                loss = objective(preds, labels)
                preds = pred_fn(preds)
                metrics.update(preds, labels)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()*preds.size(dim=0)
                ntr += preds.size(dim=0)
                # progress.console.print("[yellow]loss[/yellow]: {0:.3f}, [yellow]acc[/yellow]: {0:.3f}".format(loss.item()/preds.size(dim=0), metrics.computes().item())) 
                progress.advance(task2)

            tr_loss = tr_loss / ntr 
            tr_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task2)

            va_loss = 0
            nva = 0

            # validation
            with torch.no_grad():
                for bi, d in enumerate(va_loader):
                    _, _, mfgs = d
                    inputs = mfgs[0].srcdata["feat"]
                    labels = mfgs[-1].dstdata["label"]
                    preds = model(mfgs, inputs)
                    loss = objective(preds, labels)
                    preds = pred_fn(preds)
                    metrics.update(preds, labels)
                    va_loss += loss.item()*preds.size(dim=0)
                    nva += preds.size(dim=0)
                    # progress.console.print("[yellow]va_loss[/yellow]: {0:.3f}, [yellow]va_acc[/yellow]: {0:.3f}".format(loss.item()/preds.size(dim=0), metrics.computes().item())) 
                    progress.advance(task3)

            if (args.debug == 1) and (epoch % int(0.2*args.epochs) == 0):
                img_conf = draw_conf(graph=sha_g, model=model, path=args.res_path + f"{name_pos}-shapos.pkl", device=device, name_plot=f'{name_pos}_{indx+1}')
                img_grad, img_loss = draw_loss_grad(graph=sha_g, model=model, path=args.res_path + f"{name_pos}-shapos.pkl", device=device, name_plot=f'{name_pos}_{indx+1}')
                
                img_conf_full, img_grad_full, img_loss_full = draw_full(tar_g=tar_g, sha_g=sha_g, model=model, path=args.res_path + f"{name_pos}-full.pkl", device=device, name_plot=f'{name_pos}_{indx+1}')

                image_conf.append(img_conf)
                image_grad.append(img_grad)
                image_loss.append(img_loss)

                image_conf_full.append(img_conf_full)
                image_grad_full.append(img_grad_full)
                image_loss_full.append(img_loss_full)
                indx += 1
                print(f"len of image confidence: {len(image_conf)}, len of image grad: {len(image_grad)}, and len of image loss: {len(image_loss)}")

            va_loss = va_loss / nva 
            va_perf = metrics.compute().item()
            metrics.reset()   
            progress.reset(task3)

            results = {
                "Target train/loss": tr_loss, 
                f"Target train/acc": tr_perf, 
                "Target val/loss": va_loss, 
                f"Target val/acc": va_perf,
            }
            history['train_history_loss'].append(tr_loss)
            history['train_history_acc'].append(tr_perf)
            history['val_history_loss'].append(va_loss)
            history['val_history_acc'].append(va_perf)
            es(epoch=epoch, epoch_score=tr_loss, model=model, model_path=args.model_path + model_name)
            tracker_log(dct = results)
            progress.console.print(f"Epoch {epoch}: [yellow]loss[/yellow]: {tr_loss}, [yellow]acc[/yellow]: {tr_perf}, [yellow]va_loss[/yellow]: {va_loss}, [yellow]va_acc[/yellow]: {va_perf}") 
            progress.advance(task1)

        if args.debug == 1:
            log_images(images=image_conf, mode='Confidence')
            log_images(images=image_grad, mode="Grad norm")
            log_images(images=image_loss, mode="Loss")

            log_images(images=image_conf_full, mode='Confidence Full')
            log_images(images=image_grad_full, mode="Grad norm Full")
            log_images(images=image_loss_full, mode="Loss Full")

        console.log(f"Done Training target model: :white_check_mark:")
    model.load_state_dict(torch.load(args.model_path + model_name))
    return model, history

def evaluate(args, te_loader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device, history:Dict):

    if args.num_class > 1:
        objective = torch.nn.CrossEntropyLoss().to(device)
        pred_fn = torch.nn.Softmax(dim=1).to(device)
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    else:
        objective = torch.nn.BCEWithLogitsLoss().to(device)
        pred_fn = torch.nn.Sigmoid().to(device)
        metrics = torchmetrics.classification.BinaryAccuracy().to(device)
    
    console.log(f"[green]Evaluate Test / Objective of the training process[/green]: {objective}")
    console.log(f"[green]Evaluate Test / Predictive activation[/green]: {pred_fn}")
    console.log(f"[green]Evaluate Test / Evaluating with metrics[/green]: {metrics}")

    with Progress(console=console) as progress:
        task1 = progress.add_task("[red]Evaluating Test ...", total=len(te_loader))
        te_loss = 0
        nte = 0
        # validation
        with torch.no_grad():
            for bi, d in enumerate(te_loader):
                _, _, mfgs = d
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                preds = model(mfgs, inputs)
                loss = objective(preds, labels)
                preds = pred_fn(preds)
                metrics.update(preds, labels)
                te_loss += loss.item()
                nte += preds.size(dim=0)
                progress.update(task1, advance=bi+1)

            te_loss = te_loss / nte 
            te_perf = metrics.compute().item()
            wandb.run.summary['te_loss'] = '{0:.3f}'.format(te_loss)
            wandb.run.summary['te_acc'] = '{0:.3f}'.format(te_perf)
            history['best_test_loss'] = te_loss
            history['best_test_perf'] = te_perf
            metrics.reset()

def log_images(images:list, mode:str):
    print(f"Logging for mode {mode}: len of images is {len(images)}")
    columns=[f'{mode}_at_epoch_{i}' for i in range(len(images))]
    data = [[wandb.Image(img) for img in images]]
    print(columns, data)
    test_table = wandb.Table(data=data, columns=columns)
    wandb.log({f"{mode} of shadow graph" : test_table})