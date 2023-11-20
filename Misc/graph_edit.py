import torch

def get_nr_vertices(data):
    if data.x is not None:
            return data.x.shape[0]
    else:
        return int(torch.max(data.edge_index)) + 1

def add_undir_edge(edge_index, v1, v2):
    return torch.cat((edge_index, torch.tensor([[v1], [v2]]), torch.tensor([[v2], [v1]])), 1)

def add_dir_edge(edge_index, v_from, v_to):
    return torch.cat((edge_index, torch.tensor([[v_from], [v_to]])), 1)