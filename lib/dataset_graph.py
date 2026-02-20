import torch
from torch_geometric.data import Data


def get_individual_graph(conn, abs_flag, topk_ratio):
    """
    input: conn (torch.Tensor), topk_ratio
    output: PyG Data
    """

    # if conn.shape != (268, 268):
    #     raise ValueError('Size error.')

    if abs_flag == 1:
        adj = torch.abs(conn)
    else:
        adj = conn

    upper_triangle_mask = torch.triu(torch.ones_like(adj, dtype=torch.bool), diagonal=1)
    adj = torch.where(upper_triangle_mask, adj, torch.zeros_like(adj))

    threshold = torch.quantile(adj[upper_triangle_mask], 1 - topk_ratio)
    adj[adj < threshold] = 0

    rows, cols = torch.where(adj > 0)
    edge_index = torch.stack([rows, cols], dim=0).type(torch.long)
    edge_weight = conn[rows, cols].type(torch.float)

    node_features = conn.type(torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)