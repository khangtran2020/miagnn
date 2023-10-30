import os
import dgl
import torch
import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from rich.pretty import pretty_repr
from Utils.console import console
from Utils.utils import read_pickel, save_dict
from Models.model import GraphSAGE, GAT

class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                console.log('Validation score improved ({self.val_score} --> {epoch_score}). Saving model!', style='info')
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def init_model(args):
    model = None
    if args.model == 'sage':
        model = GraphSAGE(in_feats=args.num_feat, n_hidden=args.hdim, n_classes=args.num_class,
                          n_layers=args.nlay, dropout=args.dout, aggregator_type=args.aggtype)
    elif args.model == 'gat':
        model = GAT(in_feats=args.num_feat, n_hidden=args.hdim, n_classes=args.num_class, n_layers=args.nlay,
                    num_head=args.nhead, dropout=args.dout)
    return model

def draw_conf(graph:dgl.DGLGraph, model:torch.nn.Module, path:str, device:torch.device, name_plot:str):

    pred_fn = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        model.to(device)
        preds = pred_fn(model.full(g=graph.to(device), x=graph.ndata['feat'].to(device))).cpu()
        log_p = torch.log(preds + 1e-12)
        conf = torch.sum(-1*preds*log_p, dim=1)

    conf = conf.numpy()

    pos_mask_tr = graph.ndata['pos_mask_tr']
    neg_mask_tr = graph.ndata['neg_mask_tr']

    pos_mask_te = graph.ndata['pos_mask_te']
    neg_mask_te = graph.ndata['neg_mask_te']


    id_postr = (pos_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()
    id_negtr = (neg_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()

    id_poste = (pos_mask_te == 1).nonzero(as_tuple=True)[0].tolist()
    id_negte = (neg_mask_te == 1).nonzero(as_tuple=True)[0].tolist()


    G = graph.to_networkx()
    if os.path.exists(path=path):
        pos = read_pickel(path)
    else:
        pos = nx.spring_layout(G)
        save_dict(path=path, dct=pos)

    plt.figure(num=None, figsize=(15, 15))
    # plt.axis('off')
    cmap=plt.cm.Blues
    vmin = min(conf)
    vmax = max(conf)

    nx.draw_networkx_nodes(G,pos,nodelist=id_postr, node_color=conf[id_postr], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negtr, node_color=conf[id_negtr], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_poste, node_color=conf[id_poste], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negte, node_color=conf[id_negte], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G,pos,arrows=True)
    # nx.draw_networkx_labels(G,pos)
    plt.savefig(f"results/dict/{name_plot}_conf.jpg", bbox_inches='tight')

    img = Image.open(f"results/dict/{name_plot}_conf.jpg")
    img_arr = np.array(img)
    return img_arr

def draw_loss_grad(graph:dgl.DGLGraph, model:torch.nn.Module, path:str, device:torch.device, name_plot:str):

    model.to(device)
    criter = torch.nn.CrossEntropyLoss(reduction='none')
    preds = model.full(g=graph.to(device), x=graph.ndata['feat'].to(device))
    target = graph.ndata['label'].to(device)
    losses = criter(preds, target)
    model.zero_grad()

    grad_overall = torch.Tensor([]).to(device)
    loss_overall = []
    for los in losses:
        
        los.backward(retain_graph=True)
        grad_sh = torch.Tensor([]).to(device)

        for name, p in model.named_parameters():
            if p.grad is not None:
                new_grad = p.grad.detach().clone()
                grad_sh = torch.cat((grad_sh, new_grad.flatten()), dim=0)
        model.zero_grad()
        grad_sh = torch.unsqueeze(grad_sh, dim=0)
        grad_overall = torch.cat((grad_overall, grad_sh), dim=0)
        loss_overall.append(los.detach().item())

    grad_norm = grad_overall.detach().norm(p=2, dim=1).cpu()
    grad_norm = grad_norm.numpy()
    loss_overall = np.array(loss_overall)

    pos_mask_tr = graph.ndata['pos_mask_tr']
    neg_mask_tr = graph.ndata['neg_mask_tr']

    pos_mask_te = graph.ndata['pos_mask_te']
    neg_mask_te = graph.ndata['neg_mask_te']


    id_postr = (pos_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()
    id_negtr = (neg_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()

    id_poste = (pos_mask_te == 1).nonzero(as_tuple=True)[0].tolist()
    id_negte = (neg_mask_te == 1).nonzero(as_tuple=True)[0].tolist()


    G = graph.to_networkx()
    if os.path.exists(path=path):
        pos = read_pickel(path)
    else:
        pos = nx.spring_layout(G)
        save_dict(path=path, dct=pos)

    plt.figure(num=None, figsize=(15, 15))
    cmap=plt.cm.Greens   
    vmin = min(grad_norm)
    vmax = max(grad_norm)

    nx.draw_networkx_nodes(G,pos,nodelist=id_postr, node_color=grad_norm[id_postr], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negtr, node_color=grad_norm[id_negtr], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_poste, node_color=grad_norm[id_poste], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negte, node_color=grad_norm[id_negte], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G,pos,arrows=True)
    # nx.draw_networkx_labels(G,pos)
    plt.savefig(f"results/dict/{name_plot}_grad.jpg", bbox_inches='tight')

    img_grad = Image.open(f"results/dict/{name_plot}_grad.jpg")
    img_grad = np.array(img_grad)

    plt.figure(num=None, figsize=(15, 15))
    cmap=plt.cm.Reds
    vmin = min(loss_overall)
    vmax = max(loss_overall)

    nx.draw_networkx_nodes(G,pos,nodelist=id_postr, node_color=loss_overall[id_postr], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negtr, node_color=loss_overall[id_negtr], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_poste, node_color=loss_overall[id_poste], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negte, node_color=loss_overall[id_negte], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G,pos,arrows=True)
    # nx.draw_networkx_labels(G,pos)
    plt.savefig(f"results/dict/{name_plot}_loss.jpg", bbox_inches='tight')

    img_loss = Image.open(f"results/dict/{name_plot}_loss.jpg")
    img_loss = np.array(img_loss)
    return img_grad, img_loss

def draw_full(tar_g:dgl.DGLGraph, sha_g:dgl.DGLGraph, model:torch.nn.Module, path:str, device:torch.device, name_plot:str):

    model.to(device)
    criter = torch.nn.CrossEntropyLoss(reduction='none')
    pred_fn = torch.nn.Softmax(dim=1)

    # target graph
    tar_label = tar_g.ndata['label'].to(device)
    tar_preds = model.full(g=tar_g.to(device), x=tar_g.ndata['feat'].to(device))
    tar_losses = criter(tar_preds, tar_label)
    model.zero_grad()

    tar_grad = torch.Tensor([]).to(device)
    for los in tar_losses:
        
        los.backward(retain_graph=True)
        grad_sh = torch.Tensor([]).to(device)

        for name, p in model.named_parameters():
            if p.grad is not None:
                new_grad = p.grad.detach().clone()
                grad_sh = torch.cat((grad_sh, new_grad.flatten()), dim=0)
        model.zero_grad()
        grad_sh = torch.unsqueeze(grad_sh, dim=0)
        tar_grad = torch.cat((tar_grad, grad_sh), dim=0)
    
    tar_grad = tar_grad.detach().norm(p=2, dim=1).cpu().numpy()
    tar_losses = tar_losses.detach().cpu().numpy()

    tar_preds = pred_fn(tar_preds.detach())
    log_tar_preds = torch.log(tar_preds + 1e-12)
    tar_conf = torch.sum(-1*tar_preds*log_tar_preds, dim=1).cpu().numpy()

    # shadow graph
    sha_label = sha_g.ndata['label'].to(device)
    sha_preds = model.full(g=sha_g.to(device), x=sha_g.ndata['feat'].to(device))
    sha_losses = criter(sha_preds, sha_label)
    model.zero_grad()

    sha_grad = torch.Tensor([]).to(device)
    for los in sha_losses:
        
        los.backward(retain_graph=True)
        grad_sh = torch.Tensor([]).to(device)

        for name, p in model.named_parameters():
            if p.grad is not None:
                new_grad = p.grad.detach().clone()
                grad_sh = torch.cat((grad_sh, new_grad.flatten()), dim=0)
        model.zero_grad()
        grad_sh = torch.unsqueeze(grad_sh, dim=0)
        sha_grad = torch.cat((sha_grad, grad_sh), dim=0)

    sha_grad = sha_grad.detach().norm(p=2, dim=1).cpu().numpy()
    sha_losses = sha_losses.detach().cpu().numpy()

    sha_preds = pred_fn(sha_preds.detach())
    log_sha_preds = torch.log(sha_preds + 1e-12)
    sha_conf = torch.sum(-1*sha_preds*log_sha_preds, dim=1).cpu().numpy()

    # getting max and min for loss / grad / conf
    lmin = min(min(tar_losses), min(sha_losses))
    lmax = max(max(tar_losses), max(sha_losses))

    gmin = min(min(tar_grad), min(sha_grad))
    gmax = max(max(tar_grad), max(sha_grad))

    cmin = min(min(tar_conf), min(sha_conf))
    cmax = max(max(tar_conf), max(sha_conf))

    # getting position
    if os.path.exists(path=path):
        pos_mask = tar_g.ndata['tr_mask']
        neg_mask = tar_g.ndata['te_mask']
        id_pos_tar = (pos_mask == 1).nonzero(as_tuple=True)[0].tolist()
        id_neg_tar = (neg_mask == 1).nonzero(as_tuple=True)[0].tolist()

        pos_mask = sha_g.ndata['tr_mask']
        neg_mask = sha_g.ndata['te_mask']
        id_pos_sha = (pos_mask == 1).nonzero(as_tuple=True)[0].tolist()
        id_neg_sha = (neg_mask == 1).nonzero(as_tuple=True)[0].tolist()
        pos = read_pickel(path)
    else:

        # target graph
        pos_mask = tar_g.ndata['tr_mask']
        neg_mask = tar_g.ndata['te_mask']
        id_pos_tar = (pos_mask == 1).nonzero(as_tuple=True)[0].tolist()
        id_neg_tar = (neg_mask == 1).nonzero(as_tuple=True)[0].tolist()

        pos_pos = np.random.normal(loc=5.0, scale=1.0, size=(len(id_pos_tar), 2))
        pos_pos = list(zip(pos_pos[:,0], pos_pos[:,1]))
        pos_tar_pos = dict(zip(id_pos_tar, pos_pos))


        pos_neg = np.random.normal(loc=2.0, scale=0.5, size=(len(id_neg_tar), 2))
        pos_neg = list(zip(pos_neg[:,0], pos_neg[:,1]))
        pos_tar_neg = dict(zip(id_neg_tar, pos_neg))

        print(f"Target pos:",pretty_repr(pos_tar_pos))
        print(f"Target neg:",pretty_repr(pos_tar_neg))

        pos_tar = pos_tar_pos.update(pos_tar_neg)

        # shadow graph
        pos_mask = sha_g.ndata['tr_mask']
        neg_mask = sha_g.ndata['te_mask']
        id_pos_sha = (pos_mask == 1).nonzero(as_tuple=True)[0].tolist()
        id_neg_sha = (neg_mask == 1).nonzero(as_tuple=True)[0].tolist()

        pos_pos = np.random.normal(loc=-1.0, scale=0.5, size=(len(id_pos_sha), 2))
        pos_pos = list(zip(pos_pos[:,0], pos_pos[:,1]))
        pos_sha_pos = dict(zip(id_pos_sha, pos_pos))


        pos_neg = np.random.normal(loc=-2.0, scale=0.5, size=(len(id_neg_sha), 2))
        pos_neg = list(zip(pos_neg[:,0], pos_neg[:,1]))
        pos_sha_neg = dict(zip(id_neg_sha, pos_neg))

        print(f"Shadow pos:",pretty_repr(pos_sha_pos))
        print(f"Shadow neg:",pretty_repr(pos_sha_neg))

        pos_sha = pos_sha_pos.update(pos_sha_neg)

        pos = {
            'tar': pos_tar,
            'sha': pos_sha
        }
        save_dict(path=path, dct=pos)

    G_tar = tar_g.to_networkx()
    G_sha = sha_g.to_networkx()

    # plot grad norm
    plt.figure(num=None, figsize=(20, 20))
    cmap=plt.cm.Greens

    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_pos_tar, node_color=tar_grad[id_pos_tar], cmap=cmap, node_shape='o', vmin=gmin, vmax=gmax)
    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_neg_tar, node_color=tar_grad[id_neg_tar], cmap=cmap, node_shape='s', vmin=gmin, vmax=gmax)
    nx.draw_networkx_edges(G_tar,pos['tar'],arrows=True)

    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_pos_sha, node_color=sha_grad[id_pos_sha], cmap=cmap, node_shape='o', vmin=gmin, vmax=gmax)
    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_neg_sha, node_color=sha_grad[id_neg_sha], cmap=cmap, node_shape='s', vmin=gmin, vmax=gmax)
    nx.draw_networkx_edges(G_sha,pos['sha'],arrows=True)
    plt.savefig(f"results/dict/{name_plot}-grad-full.jpg", bbox_inches='tight')
    img_grad = Image.open(f"results/dict/{name_plot}-grad-full.jpg")
    img_grad = np.array(img_grad)

    # plot loss
    plt.figure(num=None, figsize=(20, 20))
    cmap=plt.cm.Reds

    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_pos_tar, node_color=tar_losses[id_pos_tar], cmap=cmap, node_shape='o', vmin=lmin, vmax=lmax)
    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_neg_tar, node_color=tar_losses[id_neg_tar], cmap=cmap, node_shape='s', vmin=lmin, vmax=lmax)
    nx.draw_networkx_edges(G_tar,pos['tar'],arrows=True)

    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_pos_sha, node_color=sha_losses[id_pos_sha], cmap=cmap, node_shape='o', vmin=lmin, vmax=lmax)
    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_neg_sha, node_color=sha_losses[id_neg_sha], cmap=cmap, node_shape='s', vmin=lmin, vmax=lmax)
    nx.draw_networkx_edges(G_sha,pos['sha'],arrows=True)
    plt.savefig(f"results/dict/{name_plot}-loss-full.jpg", bbox_inches='tight')
    img_loss = Image.open(f"results/dict/{name_plot}-loss-full.jpg")
    img_loss = np.array(img_loss)

    plt.figure(num=None, figsize=(20, 20))
    cmap=plt.cm.Blues

    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_pos_tar, node_color=tar_conf[id_pos_tar], cmap=cmap, node_shape='o', vmin=cmin, vmax=cmax)
    nx.draw_networkx_nodes(G_tar,pos['tar'],nodelist=id_neg_tar, node_color=tar_conf[id_neg_tar], cmap=cmap, node_shape='s', vmin=cmin, vmax=cmax)
    nx.draw_networkx_edges(G_tar,pos['tar'],arrows=True)

    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_pos_sha, node_color=sha_conf[id_pos_sha], cmap=cmap, node_shape='o', vmin=cmin, vmax=cmax)
    nx.draw_networkx_nodes(G_sha,pos['sha'],nodelist=id_neg_sha, node_color=sha_conf[id_neg_sha], cmap=cmap, node_shape='s', vmin=cmin, vmax=cmax)
    nx.draw_networkx_edges(G_sha,pos['sha'],arrows=True)
    plt.savefig(f"results/dict/{name_plot}-conf-full.jpg", bbox_inches='tight')
    img_conf = Image.open(f"results/dict/{name_plot}-conf-full.jpg")
    img_conf = np.array(img_conf)

    return img_conf, img_grad, img_loss
