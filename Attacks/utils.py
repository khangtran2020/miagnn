import dgl
import torch
from Utils.utils import get_index_by_value

def generate_nohop_graph(graph:dgl.DGLGraph):
    nodes = graph.nodes().tolist()
    g = dgl.graph((nodes, nodes), num_nodes=len(nodes))
    for key in graph.ndata.keys():
        g.ndata[key] = graph.ndata[key].clone()
    return g

def init_shadow_loader(args, device: torch.device, graph:dgl.DGLGraph):

    tr_nid = get_index_by_value(a=graph.ndata['str_mask'], val=1).to(device)
    te_nid = get_index_by_value(a=graph.ndata['ste_mask'], val=1).to(device)

    sampler = dgl.dataloading.NeighborSampler([-1 for i in range(args.n_layers)])

    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nid.to(device), sampler, device=device,
                                           batch_size=args.att_bs, shuffle=True, drop_last=True)

    te_loader = dgl.dataloading.DataLoader(graph.to(device), te_nid.to(device), sampler, device=device,
                                           batch_size=args.att_bs, shuffle=False, drop_last=False)
    
    return tr_loader, te_loader