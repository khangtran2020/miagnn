import dgl
import torch
import wandb
from copy import deepcopy
from rich.table import Table
from functools import partial
from typing import Dict
from Utils.console import console, log_table
from Utils.tracking import tracker_log_table
from Utils.utils import get_index_by_value, get_index_by_list
from Data.dataset import Facebook, Arxiv
from Data.utils import node_split, filter_class_by_count, reduce_desity, \
    graph_split, get_shag_edge_info, percentage_pos, remove_edge, \
    check_overlap

def read_data(args, history, exist=False):
        
    info = {}

    graph, list_of_label = get_graph(data_name=args.data)
    graph = dgl.remove_self_loop(graph)
    graph.ndata['org_id'] = graph.nodes().clone()

    if args.data_mode == 'density':
        graph = reduce_desity(g=graph, dens_reduction=args.dens)
        x = graph.ndata['feat']
        y = graph.ndata['label']
        nodes = graph.nodes()
        src_edge, dst_edge = graph.edges()
        prop_reduced = {
            '# nodes': nodes.size(dim=0),
            '# edges': int(src_edge.size(dim=0) / 2),
            'Average degree': graph.in_degrees().float().mean().item(),
            'Node homophily': dgl.node_homophily(graph=graph, y=y),
            '# features': x.size(dim=1),
            '# labels': y.max().item() + 1 
        }
        info['reduced_graph'] = prop_reduced
        console.log(f"Done reducing density: :white_check_mark:")

    x = graph.ndata['feat']
    y = graph.ndata['label']
    nodes = graph.nodes()
    src_edge, dst_edge = graph.edges()

    prop_org = {
        '# nodes': f'{nodes.size(dim=0)}',
        '# edges': f'{int(src_edge.size(dim=0) / 2)}',
        'Average degree': f'{graph.in_degrees().float().mean().item()}',
        'Node homophily': f'{dgl.node_homophily(graph=graph, y=y)}',
        '# features':f'{ x.size(dim=1)}',
        '# labels': f'{y.max().item() + 1 }'
    }
    info['original_graph'] = prop_org
    console.log(f"Done getting graph: :white_check_mark:")

    if exist == False:
        graph = node_split(args=args, graph=graph, val_size=0.16, test_size=0.2)
        history['tr_id'] = graph.ndata['tr_mask'].tolist()
        history['va_id'] = graph.ndata['va_mask'].tolist()
        history['te_id'] = graph.ndata['te_mask'].tolist()
        history['sh_id'] = graph.ndata['sh_mask'].tolist()
    else:
        id_tr = history['tr_id']
        id_va = history['va_id']
        id_te = history['te_id']
        id_sh = history['sh_id']

        graph.ndata['tr_mask'] = torch.LongTensor(id_tr)
        graph.ndata['va_mask'] = torch.LongTensor(id_va)
        graph.ndata['te_mask'] = torch.LongTensor(id_te)
        graph.ndata['sh_mask'] = torch.LongTensor(id_sh)

    tar_g, sha_g = remove_edge(graph=graph, mode=args.gen_submode, debug=args.debug)
    if args.debug == 1:
        check_overlap(graph=tar_g, mode='target')
    console.log(f"Done splitting train/val/test: :white_check_mark:")

    args.num_class = len(list_of_label)
    args.num_feat = x.size(dim=1)

    if args.gen_submode == 'ind':
        g_tr, g_va, g_te = graph_split(graph=tar_g)
        x = g_tr.ndata['feat']
        y = g_tr.ndata['label']
        nodes = g_tr.nodes()
        src_edge, dst_edge = g_tr.edges()
        prop_train = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_tr.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_tr, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }

        x = g_va.ndata['feat']
        y = g_va.ndata['label']
        nodes = g_va.nodes()
        src_edge, dst_edge = g_va.edges()
        prop_val = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_va.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_va, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }

        x = g_te.ndata['feat']
        y = g_te.ndata['label']
        nodes = g_te.nodes()
        src_edge, dst_edge = g_te.edges()
        prop_test = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_te.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_te, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }
        
        info['train_graph'] = prop_train
        info['val_graph'] = prop_val
        info['test_graph'] = prop_test
        del x, y, nodes, src_edge, dst_edge
        del g_tr, g_va, g_te
        console.log(f"Done graph splitting: :white_check_mark:")
    else:
        console.log(f"Running [green]transductive[/green] setting -> does not need to split graph: :white_check_mark:")

    table = Table(title=f'Info of dataset: {args.data}')
    table.add_column("Graph", style="key")
    table.add_column("Property", style="key")
    table.add_column("Values", style="value")

    columns=["Graph", "Property", "Values"]
    my_data = []

    for key in info.keys():
        dct = info[key]
        num_sub_key = len(dct.keys())
        for i, subkey in enumerate(dct.keys()):
            my_data.append([f'{key}', f'{subkey}', f'{dct[subkey]}'])
            if i == 0:
                table.add_row(f'{key}', f'{subkey}', f'{dct[subkey]}')
            elif i == num_sub_key - 1:
                table.add_row('', f'{subkey}', f'{dct[subkey]}', end_section=True)
            else:
                table.add_row('', f'{subkey}', f'{dct[subkey]}')
    console.log(table)
    wandb_table = wandb.Table(data=my_data, columns=columns)
    wandb.run.log({f"Info of dataset: {args.data}": wandb_table})
    return tar_g, sha_g  

def get_graph(data_name:str):

    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        min_count = 10000
    elif data_name == 'cora':
        data = dgl.data.CoraGraphDataset()
        graph = data[0]
        min_count = 0
    elif data_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
        graph = data[0]
        min_count = 0
    elif data_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
        graph = data[0]
        min_count = 0
    elif data_name == 'facebook':
        load_data = partial(Facebook, name='UIllinois20', target='year')
        data = load_data(root='Data/datasets/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        min_count=1000
    elif data_name == 'arxiv':
        load_data = partial(Arxiv)
        data = load_data(root='Data/datasets/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        min_count=6000
    
    graph, list_of_label = filter_class_by_count(graph=graph, min_count=min_count)
    return graph, list_of_label

def blackbox_split(args, graph:dgl.DGLGraph, history=None, exist=False, mode='joint'):

    if exist == False:
        y = graph.ndata['label']
        num_classes = args.num_class

        train_mask = torch.zeros_like(y)
        test_mask = torch.zeros_like(y)

        for c in range(num_classes):
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            num_tr = int(0.5*idx.size(dim=0))
            idx = idx[torch.randperm(idx.size(dim=0))]
            train_mask[idx[:num_tr]] = True
            test_mask[idx[num_tr:]] = True

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask

        tr_idx = get_index_by_value(a=train_mask, val=1).tolist()
        te_idx = get_index_by_value(a=test_mask, val=1).tolist()
        history['sha_tr'] = tr_idx
        history['sha_te'] = te_idx
    else:
        train_mask = torch.zeros(graph.nodes().size(dim=0))
        test_mask = torch.zeros(graph.nodes().size(dim=0))
        train_mask[history['sha_tr']] = 1
        test_mask[history['sha_te']] = 1
        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask

    if mode != 'joint':
        num_node = graph.nodes().size(dim=0)
        id_tr = graph.ndata['str_mask']
        id_te = graph.ndata['ste_mask']
        src_edges, dst_edges = graph.edges()

        src_intr = id_tr[src_edges]
        src_inte = id_te[src_edges]

        dst_intr = id_tr[dst_edges]
        dst_inte = id_te[dst_edges]
        same_intr = torch.logical_and(src_intr, dst_intr)
        same_inte = torch.logical_and(src_inte, dst_inte)
        edge_mask_tar = torch.logical_or(same_intr, same_inte)

        eid_tar = get_index_by_value(a=edge_mask_tar, val=1)
        src_tar = src_edges[eid_tar]
        dst_tar = dst_edges[eid_tar]
        temp_g = dgl.graph((src_tar, dst_tar), num_nodes=num_node)
        for key in graph.ndata.keys():
            temp_g.ndata[key] = graph.ndata[key].clone()
        graph = deepcopy(temp_g)

    y = graph.ndata['label']
    src_edge, _ = graph.edges()
    prop_dict = {
        '# nodes': f'{train_mask.size(dim=0)}',
        '# training nodes': f'{train_mask.sum().item()}',
        '# testing nodes': f'{test_mask.sum().item()}',
        '# edges': f'{int(src_edge.size(dim=0) / 2)}',
        'Average degree': f'{graph.in_degrees().float().mean().item()}',
        'Node homophily': f'{dgl.node_homophily(graph=graph, y=y)}',
        '# labels': f'{y.max().item() + 1}'
    }
    log_table(dct=prop_dict, name='Shadow graph of blackbox attack')
    tracker_log_table(dct=prop_dict, name='Shadow graph of blackbox attack')
    return graph

def whitebox_split(graph:dgl.DGLGraph, history:Dict, exist:bool, ratio:float, debug:bool=0):

    nodes = graph.nodes()

    print(graph.ndata['tr_mask'].sum(), graph.ndata['va_mask'].sum(), graph.ndata['te_mask'].sum())

    if exist == False:

        tr_node, _ = get_index_by_value(a=graph.ndata['tr_mask'], val=1).sort()
        te_node, _ = get_index_by_value(a=graph.ndata['te_mask'], val=1).sort()
        num_te = min(int(te_node.size(dim=0)*0.4), int(tr_node.size(dim=0)*0.4))

        perm = torch.randperm(tr_node.size(dim=0))
        sha_pos_te, _ = tr_node[perm[:num_te]].sort()
        sha_pos_tr, _ = tr_node[perm[num_te:]].sort()

        perm = torch.randperm(te_node.size(dim=0))
        sha_neg_te, _ = te_node[perm[:num_te]].sort()
        sha_neg_tr, _ = te_node[perm[num_te:]].sort()

        # check overlap
        if debug:
            num_postr_in_poste = get_index_by_list(arr=sha_pos_tr, test_arr=sha_pos_te).size(dim=0)
            num_poste_in_postr = get_index_by_list(arr=sha_pos_te, test_arr=sha_pos_tr).size(dim=0)
            num_negtr_in_negte = get_index_by_list(arr=sha_neg_tr, test_arr=sha_neg_te).size(dim=0)
            num_negte_in_negtr = get_index_by_list(arr=sha_neg_te, test_arr=sha_neg_tr).size(dim=0)
            if (num_postr_in_poste == 0) & (num_poste_in_postr == 0):
                console.log(f"[green] No overlap between train & valid:[\green] :white_check_mark:")
            else:
                console.log(f"Node overlap between train & valid: :x:\n{get_index_by_list(arr=sha_pos_tr, test_arr=sha_pos_te).size(dim=0)}\n{get_index_by_list(arr=sha_pos_te, test_arr=sha_pos_tr).size(dim=0)}")
                
            if (num_negtr_in_negte == 0) & (num_negte_in_negtr == 0):
                console.log(f"[green] No overlap between train & test:[\green] :white_check_mark:")
            else:
                console.log(f"Node overlap between train & test: :x:\n{get_index_by_list(arr=sha_neg_tr, test_arr=sha_neg_te).size(dim=0)}\n{get_index_by_list(arr=sha_neg_te, test_arr=sha_neg_tr).size(dim=0)}")
                
        # set up train and test for shadow graph
        str_mask = torch.zeros(nodes.size(dim=0))
        ste_mask = torch.zeros(nodes.size(dim=0))

        str_mask[sha_pos_tr] = 1
        str_mask[sha_neg_tr] = 1

        ste_mask[sha_pos_te] = 1
        ste_mask[sha_neg_te] = 1


        pos_mask_tr = torch.zeros(nodes.size(dim=0))
        pos_mask_te = torch.zeros(nodes.size(dim=0))

        neg_mask_tr = torch.zeros(nodes.size(dim=0))
        neg_mask_te = torch.zeros(nodes.size(dim=0))
        membership_label = torch.zeros(nodes.size(dim=0))

        pos_mask_tr[sha_pos_tr] = 1
        pos_mask_te[sha_pos_te] = 1

        neg_mask_tr[sha_neg_tr] = 1
        neg_mask_te[sha_neg_te] = 1

        membership_label[sha_pos_tr] = 1
        membership_label[sha_pos_te] = 1

        membership_label[sha_neg_tr] = -1
        membership_label[sha_neg_te] = -1

        graph.ndata['str_mask'] = str_mask
        graph.ndata['ste_mask'] = ste_mask
        graph.ndata['sha_label'] = membership_label
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te

        history['sha_tr'] = str_mask.tolist()
        history['sha_te'] = ste_mask.tolist()
        history['sha_label'] = membership_label.tolist()
        history['pos_mask_tr'] = pos_mask_tr.tolist()
        history['pos_mask_te'] = pos_mask_te.tolist()
        history['neg_mask_tr'] = neg_mask_tr.tolist()
        history['neg_mask_te'] = neg_mask_te.tolist()
    else:
        str_mask = torch.LongTensor(history['sha_tr'])
        ste_mask = torch.LongTensor(history['sha_te'])
        pos_mask_tr = torch.LongTensor(history['pos_mask_tr'])
        pos_mask_te = torch.LongTensor(history['pos_mask_te'])
        neg_mask_tr = torch.LongTensor(history['neg_mask_tr'])
        neg_mask_te = torch.LongTensor(history['neg_mask_te'])

        graph.ndata['str_mask'] = str_mask
        graph.ndata['ste_mask'] = ste_mask
        graph.ndata['sha_label'] = torch.Tensor(history['sha_label'])
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te

    # Eliminating pos-neg and neg-pos in train
    num_node = nodes.size(dim=0)
    pos_mask_tr = graph.ndata['pos_mask_tr']
    neg_mask_tr = graph.ndata['neg_mask_tr']
    src_edges, dst_edges = graph.edges()

    src_pos = pos_mask_tr[src_edges]
    src_neg = neg_mask_tr[src_edges]

    dst_pos = pos_mask_tr[dst_edges]
    dst_neg = neg_mask_tr[dst_edges]


    pos_neg = torch.logical_and(src_pos, dst_neg).int()
    neg_pos = torch.logical_and(src_neg, dst_pos).int()

    diff_edge = pos_neg + neg_pos
    print(diff_edge.unique(return_counts=True))
    eid_tar, _ = get_index_by_value(a=diff_edge, val=0).sort()
    src_tar = src_edges[eid_tar]
    dst_tar = dst_edges[eid_tar]
    temp_targ = dgl.graph((src_tar, dst_tar), num_nodes=num_node)
    for key in graph.ndata.keys():
        temp_targ.ndata[key] = graph.ndata[key].clone()
    graph = deepcopy(temp_targ)

    if (ratio >= 0.0) and (ratio < 1):
        console.log(f"Reducing the pos-neg and neg-pos edges in test with ratio: {ratio}")
        num_node = graph.nodes().size(dim=0)
        id_pos = graph.ndata['tr_mask']
        id_neg = graph.ndata['te_mask']
        src_edges, dst_edges = graph.edges()
        index = (src_edges < dst_edges).nonzero(as_tuple=True)[0]
        src_edges = src_edges[index]
        dst_edges = dst_edges[index]

        src_pos = id_pos[src_edges]
        src_neg= id_neg[src_edges]

        dst_pos = id_pos[dst_edges]
        dst_neg = id_neg[dst_edges]
        
        pos_neg = torch.logical_and(src_pos, dst_neg)
        neg_pos = torch.logical_and(src_neg, dst_pos)
        edge_mask_tar = torch.logical_or(pos_neg, neg_pos)

        eid_take_tar = get_index_by_value(a=edge_mask_tar, val=0)
        eid_temp_tar = get_index_by_value(a=edge_mask_tar, val=1)
        num_remain = int(ratio*eid_temp_tar.size(dim=0))
        perm = torch.randperm(eid_temp_tar.size(dim=0))
        eid_remain_tar = eid_temp_tar[perm[:num_remain]]
        eid_tar = torch.cat((eid_take_tar, eid_remain_tar), dim=0)
        src_tar = src_edges[eid_tar]
        dst_tar = dst_edges[eid_tar]
        src_ed = torch.cat((src_tar,dst_tar), dim=0)
        dst_ed = torch.cat((dst_tar,src_tar), dim=0)
        temp_targ = dgl.graph((src_ed, dst_ed), num_nodes=num_node)
        for key in graph.ndata.keys():
            temp_targ.ndata[key] = graph.ndata[key].clone()
        graph = deepcopy(temp_targ)
        

    y = graph.ndata['label']
    y_mem = graph.ndata['sha_label']
    nodes = graph.nodes()
    src_edge, dst_edge = graph.edges()
    einfo = get_shag_edge_info(graph=graph)

    info = {
        '# nodes': f'{nodes.size(dim=0)}',
        '# positive training nodes': f'{pos_mask_tr.sum().item()}',
        '# negative training nodes': f'{neg_mask_tr.sum().item()}',
        '# positive testing nodes': f'{pos_mask_te.sum().item()}',
        '# negative testing nodes': f'{neg_mask_te.sum().item()}',
        '# edges': f'{int(src_edge.size(dim=0) / 2)}',
    }
    for key in einfo.keys():
        info[key] = einfo[key]

    info['Average degree'] = f'{graph.in_degrees().float().mean().item()}'
    info['Node homophily'] = f'{dgl.node_homophily(graph=graph, y=y)}'
    info['Membership homophily'] = f'{dgl.node_homophily(graph=graph, y=y_mem)}'
    info['# labels'] =f'{ y.max().item() + 1}'

    per = partial(percentage_pos, graph=graph)
    percentage = []
    for node in graph.nodes():
        percentage.append(per(node))
    percentage = torch.Tensor(percentage)
    info['Average % neighbor is positive'] = f'{percentage.sum().item() / (len(percentage) + 1e-12)}'
    info['Average % neighbor is negative'] = f'{1 - percentage.sum().item() / (len(percentage) + 1e-12)}'
    log_table(dct=info, name=f'Shadow graph whitebox info')
    tracker_log_table(dct=info, name='Shadow graph of whitebox attack')
    return graph
