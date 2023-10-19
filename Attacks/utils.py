import dgl
import torch

def generate_nohop_graph(graph:dgl.DGLGraph):
    nodes = graph.nodes().tolist()
    g = dgl.graph((nodes, nodes), num_nodes=len(nodes))
    for key in graph.ndata.keys():
        g.ndata[key] = graph.ndata[key].clone()
    return g
