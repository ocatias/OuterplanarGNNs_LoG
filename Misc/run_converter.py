"""
Trains and evaluates a model a single time for given hyperparameters.
"""

import random
import time 
import os

import wandb
import torch
import numpy as np

from Exp.parser import parse_args
from Misc.config import config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.preparation import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def track_epoch(epoch, metric_name, train_result, val_result, test_result, lr):
    wandb.log({
        "Epoch": epoch,
        "Train/Loss": train_result["total_loss"],
        "Val/Loss": val_result["total_loss"],
        f"Val/{metric_name}": val_result[metric_name],
        "Test/Loss": test_result["total_loss"],
        f"Test/{metric_name}": test_result[metric_name],
        "LearningRate": lr
        })
    Exp/run_model.py
def print_progress(train_loss, val_loss, test_loss, metric_name, val_metric, test_metric):
    print(f"\tTRAIN\t loss: {train_loss:6.4f}")
    print(f"\tVAL\t loss: {val_loss:6.4f}\t  {metric_name}: {val_metric:10.4f}")
    print(f"\tTEST\t loss: {test_loss:6.4f}\t  {metric_name}: {test_metric:10.4f}")


## new stuff

def plain_bagels(loader, config={'binfile': './outerplanaritytest', 'verbose': True}):
    '''
    This is the important function

    Given a dataloader, compute for each batch individually
    1) if the graphs in the batch are outerplanar
    2)  the Hamiltonian cycles of the outerplanar blocks

    To this end, each batch is transformed to a textual format, piped to an external program 
    which pipes its results back which is then parsed and stored in the tensors (TODO).

    The function uses the (linux) executable ``outerplanaritytest`` which might need to
    be recompiled from source if your system is different.
    The source code is available at https://github.com/pwelke/GraphMiningTools
    By cloning the repository and running 
    ``make outerplanaritytest``
    on your system, the binary can be recompiled to run on your Posix system.
    '''
    import json
    import subprocess

    tuc = 0

    for batch in loader:

        graphstring = ''
        
        # graph conversion to textual input format for all graphs in the current batch
        # assumes undirected graphs
        for i in range(batch.num_graphs):
            g = batch.get_example(i)
            graphstring += f'# {i} {0} {g.num_nodes} {g.num_edges // 2}\n'
            graphstring += " ".join(['1' for _ in range(g.num_nodes)]) + '\n'
            graphstring += " ".join([f'{g.edge_index[0,i] + 1} {g.edge_index[1,i] + 1} {1}' for i in range(g.edge_index.shape[1]) if g.edge_index[0,i] < g.edge_index[1,i]]) + '\n'
        graphstring += '$\n'

        tic = time.time()
        # the actual computation of outerplanarity and Hamiltonian cycles in a subprocess
        cmd = [config['binfile'], '-']
        proc = subprocess.run(args=cmd, capture_output=True, input=graphstring.encode("utf-8"))
        toc = time.time()

        tuc += toc - tic


        # parsing of the results (directly from stdout of the process)
        jsobjects = json.loads(proc.stdout.decode("utf-8"))
        # print(jsobjects)

        # TODO: don't know, yet, how to best store this information in node or edge features
    
    if config['verbose']:
        print(f'time spent abroad: {tuc}')


# def plain_bagels_allyoucaneat(loader, config={'binfile': './outerplanaritytest', 'verbose': True}):
#     '''
#     Given a dataloader, compute for each batch individually
#     1) if the graphs in the batch are outerplanar
#     2)  the Hamiltonian cycles of the outerplanar blocks

#     To this end, all data is transformed to a textual format, piped to an external program 
#     which pipes its results back which is then parsed and stored in the tensors (TODO).
#     '''
#     import json
#     import subprocess

#     graphstring = ''

#     for batch in loader:

#         # graph conversion to textual input format for all graphs in the current batch
#         # assumes undirected graphs
#         for i in range(batch.num_graphs):
#             g = batch.get_example(i)
#             graphstring += f'# {i} {0} {g.num_nodes} {g.num_edges // 2}\n'
#             graphstring += " ".join(['1' for _ in range(g.num_nodes)]) + '\n'
#             graphstring += " ".join([f'{g.edge_index[0,i] + 1} {g.edge_index[1,i] + 1} {1}' for i in range(g.edge_index.shape[1]) if g.edge_index[0,i] < g.edge_index[1,i]]) + '\n'
    
#     graphstring += '$\n'


#     tic = time.time()

#     # the actual computation of outerplanarity and Hamiltonian cycles in a subprocess
#     cmd = [config['binfile'], '-']
#     proc = subprocess.run(args=cmd, capture_output=True, input=graphstring.encode("utf-8"))

#     toc = time.time()
#     if config['verbose']:
#         print(f'time spent abroad: {toc -  tic}')

#     # parsing of the results (directly from stdout of the process)
#     jsobjects = json.loads(proc.stdout.decode("utf-8"))
#     # print(jsobjects)

#     # TODO: don't know, yet, how to best store this information in node or edge features


    
def main(args):
    print(args)
    device = args.device
    use_tracking = args.use_tracking
    
    set_seed(args.seed)
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_vertex_features = train_loader.dataset.num_classes, train_loader.dataset.num_node_features
    
    if args.dataset.lower() == "zinc" or "ogb" in args.dataset.lower():
        num_classes = 1
   
    try:
        num_tasks = train_loader.dataset.num_tasks
    except:
        num_tasks = 1
        
    print(f"#Features: {num_vertex_features}")
    print(f"#Classes: {num_classes}")
    print(f"#Tasks: {num_tasks}")

    import time

    tic = time.time()

    plain_bagels(train_loader)

    toc = time.time()

    print(f'time for conversion and computation {toc -  tic}')

    # tic = time.time()

    # plain_bagels_allyoucaneat(train_loader)

    # toc = time.time()

    # print(f'time for conversion and computation {toc -  tic}')





def run(passed_args = None):
    args = parse_args(passed_args)
    return main(args)

if __name__ == "__main__":
    run()