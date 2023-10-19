import dgl
import torch
import wandb
from rich.table import Table
from functools import partial
from Utils.console import console, log_table
from Utils.tracking import tracker_log_table
from Utils.utils import get_index_by_value
from Data.dataset import Facebook, Arxiv
from Data.utils import node_split, filter_class_by_count, reduce_desity, \
    graph_split, drop_isolated_node, get_shag_edge_info, percentage_pos

def read_data(args, history, exist=False):
        
    info = {}

    graph, list_of_label = get_graph(data_name=args.dataset)
    graph = dgl.remove_self_loop(graph)
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

    if args.data_mode == 'density':
        graph = reduce_desity(g=graph, dens_reduction=args.density)
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


    node_split(graph=graph, val_size=0.1, test_size=0.15)
    args.num_class = len(list_of_label)
    args.num_feat = x.size(dim=1)
    graph.ndata['org_id'] = graph.nodes().clone()

    if exist == False:
        history['tr_id'] = graph.ndata['train_mask'].tolist()
        history['va_id'] = graph.ndata['val_mask'].tolist()
        history['te_id'] = graph.ndata['test_mask'].tolist()
    else:
        # rprint(f"History is {exist} to exist, assigning masks according to previous run")
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])

        id_train = history['tr_id']
        id_val = history['va_id']
        id_test = history['te_id']

        graph.ndata['train_mask'] = torch.LongTensor(id_train)
        graph.ndata['val_mask'] = torch.LongTensor(id_val)
        graph.ndata['test_mask'] = torch.LongTensor(id_test)
    console.log(f"Done splitting train/val/test: :white_check_mark:")

    if args.general_submode == 'ind':
        if (args.data_mode == 'density') and (args.density == 1.0):
            g_train, g_val, g_test = graph_split(graph=graph, drop=False)
        else:
            g_train, g_val, g_test = graph_split(graph=graph, drop=True)
        train_mask = torch.zeros(graph.nodes().size(dim=0))
        val_mask = torch.zeros(graph.nodes().size(dim=0))
        test_mask = torch.zeros(graph.nodes().size(dim=0))
        id_intr = g_train.ndata['org_id']
        id_inva = g_val.ndata['org_id']
        id_inte = g_test.ndata['org_id']
        train_mask[id_intr] = 1
        val_mask[id_inva] = 1
        test_mask[id_inte] = 1

        graph.ndata['train_mask'] = train_mask
        graph.ndata['test_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        graph.ndata['id_intr'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).long()
        graph.ndata['id_intr'][id_intr] = g_train.nodes().clone().long()

        graph.ndata['id_inte'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).long()
        graph.ndata['id_inte'][id_inte] = g_test.nodes().clone().long()

        idx = torch.cat((id_intr, id_inva, id_inte), dim=0)
        graph = graph.subgraph(torch.LongTensor(idx))
        if (args.data_mode != 'density') or (args.density != 1.0):
            graph = drop_isolated_node(graph)
        
        x = g_train.ndata['feat']
        y = g_train.ndata['label']
        nodes = g_train.nodes()
        src_edge, dst_edge = g_train.edges()

        prop_train = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_train.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_train, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }

        x = g_val.ndata['feat']
        y = g_val.ndata['label']
        nodes = g_val.nodes()
        src_edge, dst_edge = g_val.edges()

        prop_val = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_val.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_val, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }

        x = g_test.ndata['feat']
        y = g_test.ndata['label']
        nodes = g_test.nodes()
        src_edge, dst_edge = g_test.edges()

        prop_test = {
            '# nodes': f'{nodes.size(dim=0)}',
            '# edges': f'{int(src_edge.size(dim=0) / 2)}',
            'Average degree': f'{g_test.in_degrees().float().mean().item()}',
            'Node homophily': f'{dgl.node_homophily(graph=g_test, y=y)}',
            '# features':f'{x.size(dim=1)}',
            '# labels': f'{y.max().item() + 1}'
        }
        info['train_graph'] = prop_train
        info['val_graph'] = prop_val
        info['test_graph'] = prop_test
        del x, y, nodes, src_edge, dst_edge
        console.log(f"Done graph splitting: :white_check_mark:")
    else:
        console.log(f"Running [green]transductive[/green] setting -> does not need to split graph: :white_check_mark:")


    table = Table(title=f'Info of dataset: {args.dataset}')
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
    wandb.run.log({f"Info of dataset: {args.dataset}": wandb_table})

    if args.general_submode == 'ind':
        args.num_data_point = len(g_train.nodes())
        return g_train, g_val, g_test, graph
    else:
        args.num_data_point = len(graph.nodes())
        return graph  

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
    console.log(list_of_label)
    return graph, list_of_label

def blackbox_split(graph, ratio, train_ratio=0.5, history=None, exist=False):

    if exist == False:
        y = graph.ndata['label']
        num_classes = y.max().item() + 1

        train_mask = torch.zeros_like(y)
        test_mask = torch.zeros_like(y)
        shadow_mask = torch.zeros_like(y)

        for c in range(num_classes):
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            num_nodes = idx.size(0)
            num_shadow = int(ratio*num_nodes)
            num_tr = int(train_ratio*num_shadow)
            idx = idx[torch.randperm(idx.size(0))][:num_shadow]
            shadow_mask[idx] = True
            train_mask[idx[:num_tr]] = True
            test_mask[idx[num_tr:]] = True

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_mask'] = shadow_mask

        tr_idx = get_index_by_value(a=train_mask, val=1).tolist()
        te_idx = get_index_by_value(a=test_mask, val=1).tolist()
        sha_nodes = get_index_by_value(a=shadow_mask, val=1)

        history['sha_tr'] = tr_idx
        history['sha_te'] = te_idx
        history['sha_mask'] = sha_nodes.tolist()
    else:
        train_mask = torch.zeros(graph.nodes().size(dim=0))
        test_mask = torch.zeros(graph.nodes().size(dim=0))
        shadow_mask = torch.zeros(graph.nodes().size(dim=0))

        train_mask[history['sha_tr']] = 1
        test_mask[history['sha_te']] = 1
        shadow_mask[history['sha_mask']] = 1

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_mask'] = shadow_mask
        sha_nodes = get_index_by_value(a=shadow_mask, val=1)
    shadow_graph = graph.subgraph(sha_nodes)
    y = shadow_graph.ndata['label']
    src_edge, _ = shadow_graph.edges()
    prop_dict = {
        '# nodes': f'{train_mask.size(dim=0)}',
        '# training nodes': f'{train_mask.sum().item()}',
        '# testing nodes': f'{test_mask.sum().item()}',
        '# edges': f'{int(src_edge.size(dim=0) / 2)}',
        'Average degree': f'{shadow_graph.in_degrees().float().mean().item()}',
        'Node homophily': f'{dgl.node_homophily(graph=shadow_graph, y=y)}',
        '# labels': f'{y.max().item() + 1}'
    }
    log_table(dct=prop_dict, name='Shadow graph of blackbox attack')
    tracker_log_table(dct=prop_dict, name='Shadow graph of blackbox attack')
    return shadow_graph

def whitebox_split(graph, ratio, history=None, exist=False, diag=False):

    org_nodes = graph.nodes()
    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = int(ratio * tr_node.size(dim=0))
        perm = torch.randperm(tr_node.size(dim=0))
        shatr_nodes = tr_node[perm[:num_shadow]]

        num_half = min(int(te_node.size(dim=0)*0.4), int(shatr_nodes.size(dim=0)*0.4))
        # print("Half", num_half)

        perm = torch.randperm(shatr_nodes.size(dim=0))
        sha_pos_te = shatr_nodes[perm[:num_half]]
        sha_pos_tr = shatr_nodes[perm[num_half:]]

        perm = torch.randperm(te_node.size(dim=0))
        sha_neg_te = te_node[perm[:num_half]]
        sha_neg_tr = te_node[perm[num_half:]]

        train_mask = torch.zeros(org_nodes.size(dim=0))
        test_mask = torch.zeros(org_nodes.size(dim=0))

        pos_mask_tr = torch.zeros(org_nodes.size(dim=0))
        pos_mask_te = torch.zeros(org_nodes.size(dim=0))

        neg_mask_tr = torch.zeros(org_nodes.size(dim=0))
        neg_mask_te = torch.zeros(org_nodes.size(dim=0))
        
        pos_mask = torch.zeros(org_nodes.size(dim=0))
        neg_mask = torch.zeros(org_nodes.size(dim=0))

        membership_label = torch.zeros(org_nodes.size(dim=0))

        train_mask[sha_pos_tr] = 1
        train_mask[sha_neg_tr] = 1

        test_mask[sha_pos_te] = 1
        test_mask[sha_neg_te] = 1

        pos_mask_tr[sha_pos_tr] = 1
        pos_mask_te[sha_pos_te] = 1

        neg_mask_tr[sha_neg_tr] = 1
        neg_mask_te[sha_neg_te] = 1

        pos_mask[sha_pos_tr] = 1
        pos_mask[sha_pos_te] = 1

        neg_mask[sha_neg_tr] = 1
        neg_mask[sha_neg_te] = 1

        membership_label[sha_pos_tr] = 1
        membership_label[sha_pos_te] = 1

        membership_label[sha_neg_tr] = -1
        membership_label[sha_neg_te] = -1

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_label'] = membership_label
        graph.ndata['pos_mask'] = pos_mask
        graph.ndata['neg_mask'] = neg_mask
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te

        shadow_nodes = torch.cat((shatr_nodes, te_node), dim=0)

        history['sha_tr'] = train_mask.tolist()
        history['sha_te'] = test_mask.tolist()
        history['sha_label'] = membership_label.tolist()
        history['shadow_nodes'] = shadow_nodes.tolist()
        history['pos_mask'] = pos_mask.tolist()
        history['neg_mask'] = neg_mask.tolist()
        history['pos_mask_tr'] = pos_mask_tr.tolist()
        history['pos_mask_te'] = pos_mask_te.tolist()
        history['neg_mask_tr'] = neg_mask_tr.tolist()
        history['neg_mask_te'] = neg_mask_te.tolist()
    else:
        train_mask = torch.LongTensor(history['sha_tr'])
        test_mask = torch.LongTensor(history['sha_te'])
        shadow_nodes = torch.LongTensor(history['shadow_nodes'])
        pos_mask = torch.LongTensor(history['pos_mask'])
        neg_mask = torch.LongTensor(history['neg_mask'])
        pos_mask_tr = torch.LongTensor(history['pos_mask_tr'])
        pos_mask_te = torch.LongTensor(history['pos_mask_te'])
        neg_mask_tr = torch.LongTensor(history['neg_mask_tr'])
        neg_mask_te = torch.LongTensor(history['neg_mask_te'])

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_label'] = torch.Tensor(history['sha_label'])
        graph.ndata['pos_mask'] = pos_mask
        graph.ndata['neg_mask'] = neg_mask
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te
    
    shadow_graph = graph.subgraph(shadow_nodes)
    y = shadow_graph.ndata['label']
    y_mem = shadow_graph.ndata['sha_label']
    nodes = shadow_graph.nodes()
    src_edge, dst_edge = shadow_graph.edges()
    einfo = get_shag_edge_info(graph=shadow_graph)

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

    info['Average degree'] = f'{shadow_graph.in_degrees().float().mean().item()}'
    info['Node homophily'] = f'{dgl.node_homophily(graph=shadow_graph, y=y)}'
    info['Membership homophily'] = f'{dgl.node_homophily(graph=shadow_graph, y=y_mem)}'
    info['# labels'] =f'{ y.max().item() + 1}'

    per = partial(percentage_pos, graph=shadow_graph)
    percentage = []
    for node in shadow_graph.nodes():
        percentage.append(per(node))
    percentage = torch.Tensor(percentage)
    info['Average % neighbor is positive'] = f'{percentage.sum().item() / (len(percentage) + 1e-12)}'
    info['Average % neighbor is negative'] = f'{1 - percentage.sum().item() / (len(percentage) + 1e-12)}'
    log_table(dct=info, name=f'Shadow graph whitebox info')
    tracker_log_table(dct=info, name='Shadow graph of whitebox attack')
    return shadow_graph
