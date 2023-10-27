import os
import dgl
import torch
import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict
from PIL import Image
from sklearn.model_selection import train_test_split
from networkx.drawing.nx_agraph import graphviz_layout
from Utils.utils import get_index_by_value, get_index_by_list, read_pickel, save_dict
from Utils.console import console

def drop_isolated_node(graph:dgl.DGLGraph):
    mask = torch.zeros_like(graph.nodes())
    src, dst = graph.edges()
    mask[src.unique()] = 1
    mask[dst.unique()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = graph.nodes()[index]
    return graph.subgraph(torch.LongTensor(nodes_id))

def filter_class_by_count(graph:dgl.DGLGraph, min_count:int):
    target = graph.ndata['label'].clone()
    counts = target.unique(return_counts=True)[1] > min_count
    index = get_index_by_value(a=counts, val=True)
    label_dict = dict(zip(index.tolist(), range(len(index))))
    mask = target.apply_(lambda x: x in index.tolist())
    graph.ndata['label'].apply_(lambda x: label_dict[x] if x in label_dict.keys() else -1)
    nodes = get_index_by_value(a=mask, val=True)
    g = graph.subgraph(nodes)
    return g, index.tolist()

def node_split(args, graph:dgl.DGLGraph, val_size:float, test_size:float):
    
    node_id = np.arange(len(graph.nodes()))
    node_label = graph.ndata['label'].tolist()

    if args.att_mode == 'blackbox':
        id_tr, id_te, y_tr, y_te = train_test_split(node_id, node_label, test_size=test_size, stratify=node_label)
        id_tr, id_va, y_tr, y_va = train_test_split(id_tr, y_tr, test_size=val_size, stratify=y_tr)
        id_tar, id_sha, y_tar, y_sha = train_test_split(node_id, node_label, test_size=args.sha_rat, stratify=node_label)
    elif args.att_mode == 'whitebox':
        id_tr, id_te, y_tr, y_te = train_test_split(node_id, node_label, test_size=test_size, stratify=node_label)
        id_tr, id_va, y_tr, _ = train_test_split(id_tr, y_tr, test_size=val_size, stratify=y_tr)
        if args.debug == 0:
            _, id_sha, _, _ = train_test_split(id_tr, y_tr, test_size=args.sha_rat, stratify=y_tr)
            id_sha = np.concatenate((id_sha, id_te), axis=0)
        else:
            _, id_sha_tr, _, _ = train_test_split(id_tr, y_tr, test_size= (200 / len(id_tr)), stratify=y_tr)
            _, id_sha_te, _, _ = train_test_split(id_te, y_te, test_size= (200 / len(id_te)), stratify=y_te)
            id_sha = np.concatenate((id_sha_tr, id_sha_te), axis=0)
        
    tr_mask = torch.zeros(graph.nodes().size(dim=0))
    va_mask = torch.zeros(graph.nodes().size(dim=0))
    te_mask = torch.zeros(graph.nodes().size(dim=0))
    sh_mask = torch.zeros(graph.nodes().size(dim=0))

    id_tr = np.sort(id_tr)
    id_va = np.sort(id_va)
    id_te = np.sort(id_te)
    id_sha = np.sort(id_sha)

    tr_mask[id_tr] = 1
    va_mask[id_va] = 1
    te_mask[id_te] = 1
    sh_mask[id_sha] = 1

    graph.ndata['tr_mask'] = tr_mask.int()
    graph.ndata['va_mask'] = va_mask.int()
    graph.ndata['te_mask'] = te_mask.int()
    graph.ndata['sh_mask'] = sh_mask.int()

    return graph

def graph_split(graph:dgl.DGLGraph):
    tr_id = get_index_by_value(a=graph.ndata['tr_mask'], val=1)
    va_id = get_index_by_value(a=graph.ndata['va_mask'], val=1)
    te_id = get_index_by_value(a=graph.ndata['te_mask'], val=1)
    tr_g = graph.subgraph(torch.LongTensor(tr_id))
    te_g = graph.subgraph(torch.LongTensor(te_id))
    va_g = graph.subgraph(torch.LongTensor(va_id))
    return tr_g, va_g, te_g

def reduce_desity(g:dgl.DGLGraph, dens_reduction:float):
    src_edge, dst_edge = g.edges()
    index = (src_edge < dst_edge).nonzero(as_tuple=True)[0]
    src_edge = src_edge[index]
    dst_edge = dst_edge[index]

    num_edge = src_edge.size(dim=0)
    num_node = g.nodes().size(dim=0)

    dens = num_edge / num_node
    dens = dens * (1 - dens_reduction)
    num_edge_new = int(dens * num_node)
    if num_edge_new == 0:
        new_g = dgl.graph((torch.LongTensor([]), torch.LongTensor([])), num_nodes=num_node)
        for key in g.ndata.keys():
            new_g.ndata[key] = g.ndata[key].clone()
    else:
        indices = np.arange(num_edge)

        chosen_index = torch.from_numpy(np.random.choice(a=indices, size=num_edge_new, replace=False)).int()
        src_edge_new = torch.index_select(input=src_edge, dim=0, index=chosen_index)
        dst_edge_new = torch.index_select(input=dst_edge, dim=0, index=chosen_index)

        src_edge_undirected = torch.cat((src_edge_new, dst_edge_new), dim=0)
        dst_edge_undirected = torch.cat((dst_edge_new, src_edge_new), dim=0)

        new_g = dgl.graph((src_edge_undirected, dst_edge_undirected), num_nodes=num_node)
        for key in g.ndata.keys():
            new_g.ndata[key] = g.ndata[key].clone()
    return new_g

def get_shag_edge_info(graph:dgl.DGLGraph): 
    
    info = {}
    src_edge, dst_edge = graph.edges()

    # get edges in the same set in train 
    pos_mask = graph.ndata['tr_mask']
    neg_mask = graph.ndata['te_mask']

    src_edge_pos = pos_mask[src_edge]
    dst_edge_pos = pos_mask[dst_edge]

    src_edge_neg = neg_mask[src_edge]
    dst_edge_neg = neg_mask[dst_edge]

    pos_pos = torch.logical_and(src_edge_pos, dst_edge_pos).int()
    pos_neg = torch.logical_and(src_edge_pos, dst_edge_neg).int()
    neg_pos = torch.logical_and(src_edge_neg, dst_edge_pos).int()
    neg_neg = torch.logical_and(src_edge_neg, dst_edge_neg).int()
    indx_pos_pos = get_index_by_value(a=pos_pos, val=1)
    indx_pos_neg = get_index_by_value(a=pos_neg, val=1)
    indx_neg_pos = get_index_by_value(a=neg_pos, val=1)
    indx_neg_neg = get_index_by_value(a=neg_neg, val=1)

    indx_same = torch.cat((indx_pos_pos, indx_neg_neg), dim=0)
    indx_diff = torch.cat((indx_pos_neg, indx_neg_pos), dim=0)

    info['# edge pos-pos'] = f'{int(indx_pos_pos.size(dim=0)/2)}'
    info['# edge neg-neg'] = f'{int(indx_neg_neg.size(dim=0)/2)}'
    info['# edge pos-neg'] = f'{int(indx_pos_neg.size(dim=0)/2)}'
    info['# edge neg-pos'] = f'{int(indx_neg_pos.size(dim=0)/2)}'
    info['# edge same'] = f'{int(indx_same.size(dim=0)/2)}'
    info['# edge diff'] = f'{int(indx_diff.size(dim=0)/2)}'

    return info

def percentage_pos(node:torch.Tensor, graph:dgl.DGLGraph):
    frontier = graph.sample_neighbors(node, -1)
    mask = torch.zeros_like(frontier.nodes())
    src, dst = frontier.edges()
    mask[src.unique().long()] = 1
    mask[dst.unique().long()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = frontier.nodes()[index]
    num_pos = graph.ndata['tr_mask'][nodes_id.long()].sum()
    num_neg = graph.ndata['te_mask'][nodes_id.long()].sum()
    pos_percentage = num_pos.item() / (num_pos.item() + num_neg.item() + 1e-12)
    return pos_percentage

def init_loader(args, device:torch.device, graph:dgl.DGLGraph):

    tr_sampler = dgl.dataloading.NeighborSampler([args.nnei for i in range(args.nlay)], mask='target')
    te_sampler = dgl.dataloading.NeighborSampler([-1 for i in range(args.nlay)], mask='target')
    tr_nodes = get_index_by_value(a=graph.ndata['tr_mask'], val=1).to(device)
    va_nodes = get_index_by_value(a=graph.ndata['va_mask'], val=1).to(device)
    te_nodes = get_index_by_value(a=graph.ndata['te_mask'], val=1).to(device)
    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nodes, tr_sampler, device=device, batch_size=args.bs, 
                                        shuffle=True, drop_last=True)    
    va_loader = dgl.dataloading.DataLoader(graph.to(device), va_nodes, te_sampler, device=device, batch_size=args.bs, 
                                        shuffle=False, drop_last=False)
    te_loader = dgl.dataloading.DataLoader(graph.to(device), te_nodes, te_sampler, device=device, batch_size=args.bs, 
                                        shuffle=False, drop_last=False)
    return tr_loader, va_loader, te_loader

def remove_edge(graph:dgl.DGLGraph, mode:str, debug:int=0):

    num_node = graph.nodes().size(dim=0)
    sha_nodes = get_index_by_value(a=graph.ndata['sh_mask'], val=1)

    tr_nodes, _ = get_index_by_value(a=graph.ndata['tr_mask'], val=1).sort()
    va_nodes, _ = get_index_by_value(a=graph.ndata['va_mask'], val=1).sort()
    te_nodes, _ = get_index_by_value(a=graph.ndata['te_mask'], val=1).sort()
    tar_nodes, _ = torch.cat((tr_nodes, va_nodes, te_nodes), dim=0).unique().sort()
    sha_g = graph.subgraph(sha_nodes)
    tar_g = graph.subgraph(tar_nodes)
        
    if mode == 'ind':

        num_node = tar_g.nodes().size(dim=0)
        id_tr = tar_g.ndata['tr_mask']
        id_va = tar_g.ndata['va_mask']
        id_te = tar_g.ndata['te_mask']
        
        src_edges, dst_edges = tar_g.edges()
        src_intr = id_tr[src_edges]
        src_inva = id_va[src_edges]
        src_inte = id_te[src_edges]

        dst_intr = id_tr[dst_edges]
        dst_inva = id_va[dst_edges]
        dst_inte = id_te[dst_edges]

        same_intr = torch.logical_and(src_intr, dst_intr)
        same_inva = torch.logical_and(src_inva, dst_inva)
        same_inte = torch.logical_and(src_inte, dst_inte)

        idx_eintr, _ = get_index_by_value(a=same_intr, val=1).sort()
        idx_einva, _ = get_index_by_value(a=same_inva, val=1).sort()
        idx_einte, _ = get_index_by_value(a=same_inte, val=1).sort()

        if debug:
            num_tred_in_vaed = get_index_by_list(arr=idx_eintr, test_arr=idx_einva).size(dim=0)
            num_vaed_in_tred = get_index_by_list(arr=idx_einva, test_arr=idx_eintr).size(dim=0)
            num_tred_in_teed = get_index_by_list(arr=idx_eintr, test_arr=idx_einte).size(dim=0)
            num_teed_in_tred = get_index_by_list(arr=idx_einte, test_arr=idx_eintr).size(dim=0)
            num_teed_in_vaed = get_index_by_list(arr=idx_einte, test_arr=idx_einva).size(dim=0)
            num_vaed_in_teed = get_index_by_list(arr=idx_einva, test_arr=idx_einte).size(dim=0)
            
            if (num_tred_in_vaed == 0) & (num_vaed_in_tred == 0):
                console.log(f"[green] No overlap between train & valid:[\green] :white_check_mark:")
            else:
                console.log(f"Edges overlap between train & valid: :x:\n{get_index_by_list(arr=same_intr, test_arr=same_inva)}\n{get_index_by_list(arr=same_inva, test_arr=same_intr)}")
                
            if (num_tred_in_teed == 0) & (num_teed_in_tred == 0):
                console.log(f"[green] No overlap between train & test:[\green] :white_check_mark:")
            else:
                console.log(f"Edges overlap between train & test: :x:\n{get_index_by_list(arr=same_intr, test_arr=same_inte)}\n{get_index_by_list(arr=same_inte, test_arr=same_inva)}")
                
            if (num_teed_in_vaed == 0) & (num_vaed_in_teed == 0):
                console.log(f"[green] No overlap between test & valid:[\green] :white_check_mark:")
            else:
                console.log(f"Edges overlap between valid & test: :x:\n{get_index_by_list(arr=same_inte, test_arr=same_inva)}\n{get_index_by_list(arr=same_inva, test_arr=same_inte)}")
    

        id_edge, _ = torch.cat((idx_eintr, idx_einva, idx_einte), dim=0).sort()
        src_tar = src_edges[id_edge]
        dst_tar = dst_edges[id_edge]
        temp_targ = dgl.graph((src_tar, dst_tar), num_nodes=num_node)
        for key in tar_g.ndata.keys():
            temp_targ.ndata[key] = tar_g.ndata[key].clone()
        tar_g = deepcopy(temp_targ)
    return tar_g, sha_g

def check_overlap(graph:dgl.DGLGraph, mode:str):

    num_node = graph.nodes().size(dim=0)
    if mode == 'target':
        tr_nodes = get_index_by_value(a=graph.ndata['tr_mask'], val=1)
        va_nodes = get_index_by_value(a=graph.ndata['va_mask'], val=1)
        te_nodes = get_index_by_value(a=graph.ndata['te_mask'], val=1)

        num_vanode_in_trnode = get_index_by_list(arr=va_nodes, test_arr=tr_nodes).size(dim=0)
        num_trnode_in_vanode = get_index_by_list(arr=tr_nodes, test_arr=va_nodes).size(dim=0)
        num_tenode_in_trnode = get_index_by_list(arr=te_nodes, test_arr=tr_nodes).size(dim=0)
        num_trnode_in_tenode = get_index_by_list(arr=tr_nodes, test_arr=te_nodes).size(dim=0)
        num_vanode_in_tenode = get_index_by_list(arr=va_nodes, test_arr=te_nodes).size(dim=0)
        num_tenode_in_vanode = get_index_by_list(arr=te_nodes, test_arr=va_nodes).size(dim=0)

        if (num_vanode_in_trnode == 0) & (num_trnode_in_vanode == 0):
            console.log(f"[green] No overlap between train & valid:[\green] :white_check_mark:")
        else:
            console.log(f"Node overlap between train & valid: :x:\n{get_index_by_list(arr=va_nodes, test_arr=tr_nodes)}\n{get_index_by_list(arr=tr_nodes, test_arr=va_nodes)}")
            
        if (num_tenode_in_trnode == 0) & (num_trnode_in_tenode == 0):
            console.log(f"[green] No overlap between train & test:[\green] :white_check_mark:")
        else:
            console.log(f"Node overlap between train & test: :x:\n{get_index_by_list(arr=te_nodes, test_arr=tr_nodes)}\n{get_index_by_list(arr=tr_nodes, test_arr=te_nodes)}")
            
        if (num_vanode_in_tenode == 0) & (num_tenode_in_vanode == 0):
            console.log(f"[green] No overlap between test & valid:[\green] :white_check_mark:")
        else:
            console.log(f"Node overlap between valid & test: :x:\n{get_index_by_list(arr=va_nodes, test_arr=te_nodes)}\n{get_index_by_list(arr=te_nodes, test_arr=te_nodes)}")
    else:
        
        tr_nodes = get_index_by_value(a=graph.ndata['str_mask'], val=1)
        te_nodes = get_index_by_value(a=graph.ndata['ste_mask'], val=1)

        num_tenode_in_trnode = get_index_by_list(arr=te_nodes, test_arr=tr_nodes).size(dim=0)
        num_trnode_in_tenode = get_index_by_list(arr=tr_nodes, test_arr=te_nodes).size(dim=0)             
        if (num_tenode_in_trnode == 0) & (num_trnode_in_tenode == 0):
            console.log(f"[green] No overlap between train & test:[\green] :white_check_mark:")
        else:
            console.log(f"Node overlap between train & test: :x:\n{get_index_by_list(arr=te_nodes, test_arr=tr_nodes)}\n{get_index_by_list(arr=tr_nodes, test_arr=te_nodes)}")

def shadow_visualization(graph:dgl.DGLGraph, path:str):

    pos_mask_tr = graph.ndata['pos_mask_tr']
    neg_mask_tr = graph.ndata['neg_mask_tr']

    pos_mask_te = graph.ndata['pos_mask_te']
    neg_mask_te = graph.ndata['neg_mask_te']


    id_postr = (pos_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()
    id_negtr = (neg_mask_tr == 1).nonzero(as_tuple=True)[0].tolist()

    id_poste = (pos_mask_te == 1).nonzero(as_tuple=True)[0].tolist()
    id_negte = (neg_mask_te == 1).nonzero(as_tuple=True)[0].tolist()

    src_edges, dst_edges = graph.edges()
    src_pos_intr = id_postr[src_edges]
    dst_pos_intr = id_postr[dst_edges]
    pos_intr = torch.logical_and(src_pos_intr, dst_pos_intr).int()
    id_epostr, _ = get_index_by_value(a=pos_intr, val=1).sort()

    src_neg_intr = id_negtr[src_edges]
    dst_neg_intr = id_negtr[dst_edges]
    neg_intr = torch.logical_and(src_neg_intr, dst_neg_intr).int()
    id_enegtr, _ = get_index_by_value(a=neg_intr, val=1).sort()

    id_etr, _ = torch.cat((id_epostr, id_enegtr), dim=0).sort()
    
    src_etr = src_edges[id_etr].tolist()
    dst_etr = dst_edges[id_etr].tolist()
    etr_ls = list(zip(src_etr, dst_etr))



    G = graph.to_networkx()
    if os.path.exists(path=path):
        pos = read_pickel(path)
    else:
        pos = graphviz_layout(G)
        save_dict(path=path, dct=pos)

    plt.figure(num=None, figsize=(15, 15))
    plt.axis('off')

    nx.draw_networkx_nodes(G,pos,nodelist=id_postr, alpha=0.7, node_color='tab:blue', node_shape='o')
    nx.draw_networkx_nodes(G,pos,nodelist=id_negtr, alpha=0.7, node_color='tab:blue', node_shape='o')
    nx.draw_networkx_nodes(G,pos,nodelist=id_poste, alpha=0.7, node_color='tab:orange', node_shape='s')
    nx.draw_networkx_nodes(G,pos,nodelist=id_negte, alpha=0.7, node_color='tab:orange', node_shape='s')
    nx.draw_networkx_edges(G,pos,arrows=True)
    # nx.draw_networkx_labels(G,pos)
    plt.savefig("results/dict/shadow_graph.jpg", bbox_inches='tight')

    img = Image.open("results/dict/shadow_graph.jpg")
    img_arr = np.array(img)
    images = wandb.Image(
        img_arr, 
        caption="Initialize shadow graph"
        )
            
    wandb.log({"Shadow graph": images})