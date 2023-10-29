import os
import dgl
import torch
import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from networkx.drawing.nx_agraph import graphviz_layout
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

def draw_conf(graph:dgl.DGLGraph, model:torch.nn.Module, path:str, device:torch.device):

    pred_fn = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        model.to(device)
        preds = pred_fn(model.full(g=graph.to(device), x=graph.ndata['feat'].to(device))).cpu()
        log_p = torch.log(preds + 1e-12)
        conf = torch.sum(-1*preds*log_p, dim=1)

    scaler = MinMaxScaler()
    conf = conf.numpy()
    conf = scaler.fit_transform(conf.reshape(-1, 1))
    conf = np.squeeze(conf)

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
        pos = graphviz_layout(G)
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
    plt.savefig("results/dict/shadow_graph_conf.jpg", bbox_inches='tight')

    img = Image.open("results/dict/shadow_graph_conf.jpg")
    img_arr = np.array(img)
    return img_arr

def draw_grad(graph:dgl.DGLGraph, model:torch.nn.Module, path:str, device:torch.device):

    model.to(device)
    criter = torch.nn.CrossEntropyLoss(reduction='none')
    preds = model.full(g=graph.to(device), x=graph.ndata['feat'].to(device))
    target = graph.ndata['label'].to(device)
    losses = criter(preds, target)
    model.zero_grad()

    grad_overall = torch.Tensor([]).to(device)
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

    grad_norm = grad_overall.detach().norm(p=2, dim=1).cpu()
    grad_norm = grad_norm.numpy()
    scaler = MinMaxScaler()
    grad_norm = scaler.fit_transform(grad_norm.reshape(-1, 1))
    grad_norm = np.squeeze(grad_norm)

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
        pos = graphviz_layout(G)
        save_dict(path=path, dct=pos)

    plt.figure(num=None, figsize=(15, 15))
    # plt.axis('off')

    cmap=plt.cm.Reds
    vmin = min(grad_norm)
    vmax = max(grad_norm)

    nx.draw_networkx_nodes(G,pos,nodelist=id_postr, node_color=grad_norm[id_postr], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negtr, node_color=grad_norm[id_negtr], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_poste, node_color=grad_norm[id_poste], cmap=cmap, node_shape='o', vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G,pos,nodelist=id_negte, node_color=grad_norm[id_negte], cmap=cmap, node_shape='s', vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G,pos,arrows=True)
    # nx.draw_networkx_labels(G,pos)
    plt.savefig("results/dict/shadow_graph_grad.jpg", bbox_inches='tight')

    img = Image.open("results/dict/shadow_graph_grad.jpg")
    img_arr = np.array(img)
    return img_arr
