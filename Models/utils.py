import numpy as np
import torch
from Utils.console import console
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
    if args.model_type == 'sage':
        model = GraphSAGE(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                          n_layers=args.n_layers, dropout=args.dropout, aggregator_type=args.aggregator_type)
    elif args.model_type == 'gat':
        model = GAT(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class, n_layers=args.n_layers,
                    num_head=args.num_head, dropout=args.dropout)
    return model
