import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from Utils.utils import get_index_by_value

def drop_isolated_node(graph:dgl.DGLGraph):
    mask = torch.zeros_like(graph.nodes())
    src, dst = graph.edges()
    mask[src.unique()] = 1
    mask[dst.unique()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = graph.nodes()[index]
    return graph.subgraph(torch.LongTensor(nodes_id))

def filter_class_by_count(graph, min_count):
    target = deepcopy(graph.ndata['label'])
    # print(target.unique(return_counts=True)[1])
    counts = target.unique(return_counts=True)[1] > min_count
    index = get_index_by_value(a=counts, val=True)
    label_dict = dict(zip(index.tolist(), range(len(index))))
    # print("Label Dict:", label_dict)
    mask = target.apply_(lambda x: x in index.tolist())
    graph.ndata['label'].apply_(lambda x: label_dict[x] if x in label_dict.keys() else -1)
    graph.ndata['train_mask'] = graph.ndata['train_mask'] & mask
    graph.ndata['val_mask'] = graph.ndata['val_mask'] & mask
    graph.ndata['test_mask'] = graph.ndata['test_mask'] & mask
    graph.ndata['label_mask'] = mask
    return index.tolist()

def graph_split(graph:dgl.DGLGraph, drop:bool=True):
    train_id = torch.index_select(graph.nodes(), 0, graph.ndata['train_mask'].nonzero().squeeze()).numpy()
    val_id = torch.index_select(graph.nodes(), 0, graph.ndata['val_mask'].nonzero().squeeze()).numpy()
    test_id = torch.index_select(graph.nodes(), 0, graph.ndata['test_mask'].nonzero().squeeze()).numpy()
    # print(f"ORIGINAL GRAPH HAS: {graph.nodes().size()} nodes and {graph.edges()[0].size()} edges")
    train_g = graph.subgraph(torch.LongTensor(train_id))
    test_g = graph.subgraph(torch.LongTensor(test_id))
    val_g = graph.subgraph(torch.LongTensor(val_id))

    if drop == True:
        train_g = drop_isolated_node(train_g)
        val_g = drop_isolated_node(val_g)
        test_g = drop_isolated_node(test_g)

    return train_g, val_g, test_g

def node_split(graph:dgl.DGLGraph, val_size:float, test_size:float):
    node_id = np.arange(len(graph.nodes()))
    node_label = graph.ndata['label'].tolist()
    id_train, id_test, y_train, _ = train_test_split(node_id, node_label, test_size=test_size, stratify=node_label)
    id_train, id_val, _, _ = train_test_split(id_train, y_train, test_size=val_size, stratify=y_train)
    
    train_mask = torch.zeros(graph.nodes().size(dim=0))
    val_mask = torch.zeros(graph.nodes().size(dim=0))
    test_mask = torch.zeros(graph.nodes().size(dim=0))
    
    train_mask[id_train] = 1
    val_mask[id_val] = 1
    test_mask[id_test] = 1

    graph.ndata['train_mask'] = train_mask.int()
    graph.ndata['val_mask'] = val_mask.int()
    graph.ndata['test_mask'] = test_mask.int()

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
    pos_mask_tr = graph.ndata['pos_mask_tr']
    neg_mask_tr = graph.ndata['neg_mask_tr']
    src_edge_pos_intr = pos_mask_tr[src_edge]
    dst_edge_pos_intr = pos_mask_tr[dst_edge]
    mask_pos_intr = torch.logical_and(src_edge_pos_intr, dst_edge_pos_intr).int()
    indx_pos_intr = get_index_by_value(a=mask_pos_intr, val=1)
    src_edge_neg_intr = neg_mask_tr[src_edge]
    dst_edge_neg_intr = neg_mask_tr[dst_edge]
    mask_neg_intr = torch.logical_and(src_edge_neg_intr, dst_edge_neg_intr).int()
    indx_neg_intr = get_index_by_value(a=mask_neg_intr, val=1)
    indx_same_intr = torch.cat((indx_pos_intr, indx_neg_intr), dim=0)

    # get edges in diff set in train
    mask_pos_neg_intr = torch.logical_and(src_edge_pos_intr, dst_edge_neg_intr).int()
    indx_pos_neg_intr = get_index_by_value(a=mask_pos_neg_intr, val=1)
    mask_neg_pos_intr = torch.logical_and(src_edge_neg_intr, dst_edge_pos_intr).int()
    indx_neg_pos_intr = get_index_by_value(a=mask_neg_pos_intr, val=1)
    indx_diff_intr = torch.cat((indx_pos_neg_intr, indx_neg_pos_intr), dim=0)
    
    # get edges in the same set in test 
    pos_mask_te = graph.ndata['pos_mask_te']
    neg_mask_te = graph.ndata['neg_mask_te']
    src_edge_pos_inte = pos_mask_te[src_edge]
    dst_edge_pos_inte = pos_mask_te[dst_edge]
    mask_pos_inte = torch.logical_and(src_edge_pos_inte, dst_edge_pos_inte).int()
    indx_pos_inte = get_index_by_value(a=mask_pos_inte, val=1)
    src_edge_neg_inte = neg_mask_te[src_edge]
    dst_edge_neg_inte = neg_mask_te[dst_edge]
    mask_neg_inte = torch.logical_and(src_edge_neg_inte, dst_edge_neg_inte).int()
    indx_neg_inte = get_index_by_value(a=mask_neg_inte, val=1)
    indx_same_inte = torch.cat((indx_pos_inte, indx_neg_inte), dim=0)

    # get edges in diff set in test
    mask_pos_neg_inte = torch.logical_and(src_edge_pos_inte, dst_edge_neg_inte).int()
    indx_pos_neg_inte = get_index_by_value(a=mask_pos_neg_inte, val=1)
    mask_neg_pos_inte = torch.logical_and(src_edge_neg_inte, dst_edge_pos_inte).int()
    indx_neg_pos_inte = get_index_by_value(a=mask_neg_pos_inte, val=1)
    indx_diff_inte = torch.cat((indx_pos_neg_inte, indx_neg_pos_inte), dim=0)

    indx_same = torch.cat((indx_same_intr, indx_same_inte), dim=0)
    indx_diff = torch.cat((indx_diff_intr, indx_diff_inte), dim=0)

    indx_pos_pos = torch.cat((indx_pos_intr, indx_pos_inte), dim=0)
    indx_pos_neg = torch.cat((indx_pos_neg_intr, indx_pos_neg_inte), dim=0)
    indx_neg_neg = torch.cat((indx_neg_intr, indx_neg_inte), dim=0)
    indx_neg_pos = torch.cat((indx_neg_pos_intr, indx_neg_pos_inte), dim=0)

    info['# edge pos-pos'] = f'{int(indx_pos_pos.size(dim=0)/2)}'
    info['# edge neg-neg'] = f'{int(indx_neg_neg.size(dim=0)/2)}'
    info['# edge pos-neg'] = f'{int(indx_pos_neg.size(dim=0))}'
    info['# edge neg-pos'] = f'{int(indx_neg_pos.size(dim=0))}'
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
    num_pos = graph.ndata['pos_mask'][nodes_id.long()].sum()
    num_neg = graph.ndata['neg_mask'][nodes_id.long()].sum()
    pos_percentage = num_pos.item() / (num_pos.item() + num_neg.item() + 1e-12)
    return pos_percentage

def init_loader(args, device, graphs):

    tr_sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])
    te_sampler = dgl.dataloading.NeighborSampler([-1 for i in range(args.n_layers)])
    
    if args.general_submode == 'ind':
        tr_g, va_g, te_g = graphs
        tr_nodes = tr_g.nodes()
        va_nodes = va_g.nodes()
        te_nodes = te_g.nodes()
       
        tr_loader = dgl.dataloading.DataLoader(tr_g, tr_nodes, tr_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=True, drop_last=True, num_workers=0)    
        va_loader = dgl.dataloading.DataLoader(va_g, va_nodes, te_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=False, num_workers=0)
        te_loader = dgl.dataloading.DataLoader(te_g, te_nodes, te_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=False, num_workers=0)
        
    else:
        
        graph = graphs
        tr_nodes = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        va_nodes = get_index_by_value(a=graph.ndata['val_mask'], val=1)
        te_nodes = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        tr_loader = dgl.dataloading.DataLoader(graph, tr_nodes, tr_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=True, drop_last=True, num_workers=0)    
        va_loader = dgl.dataloading.DataLoader(graph, va_nodes, te_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=False, num_workers=0)
        te_loader = dgl.dataloading.DataLoader(graph, te_nodes, te_sampler, device=device, batch_size=args.batch_size, 
                                            shuffle=False, drop_last=False, num_workers=0)
        
    return tr_loader, va_loader, te_loader
