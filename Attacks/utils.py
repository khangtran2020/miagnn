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
    sampler = dgl.dataloading.NeighborSampler([-1 for i in range(args.nlay)])
    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nid.to(device), sampler, device=device,
                                        batch_size=args.att_bs, shuffle=True, drop_last=True)
    te_loader = dgl.dataloading.DataLoader(graph.to(device), te_nid.to(device), sampler, device=device,
                                        batch_size=args.att_bs, shuffle=False, drop_last=False)
    return tr_loader, te_loader

def generate_attack_samples(graph, mode, device):

    tr_mask = 'tr_mask' if mode == 'target' else 'str_mask'
    te_mask = 'te_mask' if mode == 'target' else 'ste_mask'
    pred_mask = 'pred' if mode == 'target' else 'sha_pred'
    prednh_mask = 'nh_pred' if mode == 'target' else 'shanh_pred'
    
    num_classes = graph.ndata[pred_mask].size(1)
    num_train = graph.ndata[tr_mask].sum()
    num_test = graph.ndata[te_mask].sum()
    num_half = min(num_train, num_test)

    # print(graph.ndata[pred_mask].size(), graph.ndata[prednh_mask].size())

    # labels = F.one_hot(tr_graph.ndata['label'], num_classes).float().to(device)
    org_id = graph.ndata['org_id']
    top_k, _ = torch.topk(graph.ndata[pred_mask], k=2, dim=1)
    top_k_nh, _ = torch.topk(graph.ndata[prednh_mask], k=2, dim=1)

    samples = torch.cat((top_k, top_k_nh), dim=1).to(device)

    perm = torch.randperm(num_train, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
    pos_samples = samples[idx][perm]
    org_id_pos = org_id[idx][perm]

    perm = torch.randperm(num_test, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
    neg_samples = samples[idx][perm]
    org_id_neg = org_id[idx][perm]

    # pos_entropy = Categorical(probs=pos_samples[:, :num_classes]).entropy().mean()
    # neg_entropy = Categorical(probs=neg_samples[:, :num_classes]).entropy().mean()

    # console.debug(f'pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

    org_id = torch.cat((org_id_neg, org_id_pos), dim=0)
    x = torch.cat([neg_samples, pos_samples], dim=0)
    y = torch.cat([
        torch.zeros(num_half, dtype=torch.long, device=device),
        torch.ones(num_half, dtype=torch.long, device=device),
    ])

    # shuffle data
    perm = torch.randperm(2 * num_half, device=device)
    org_id, x, y = org_id[perm], x[perm], y[perm]
    return x, y, org_id
