# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:14:49 2025

demo code for graph DMD on node classification tasks, the hyperparameter setting is for CORA only.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import  APPNP
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CitationFull
from models import MLP, GCN, GPRGNN, H2GCN, SAGE2, GCN2Conv
import argparse
import time
from torch_geometric.utils import to_undirected
import os.path as osp
from pydmd import DMD, BOPDMD,EDMD, PiDMD
from pydmd.plotter import plot_summary
import scipy.sparse as sp
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalized_adjacency_matrix(edge_index, num_nodes):
    # Add self-loops to the edge indices
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)


    # Calculate the degree matrix
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))


    # Calculate the inverse square root of the degree matrix
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0


    # Compute the normalized adjacency matrix
    row, col = edge_index
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_normalized = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))


    return adj_normalized.to_dense()


class myGCNConv(nn.Module):
    def __init__(self, num_input, num_output, num_classes, Phi, lamb_values):
        super().__init__()
        self.Phi = Phi
        self.lamb_values = lamb_values
        self.num_input = num_input
        self.num_output = num_output
        self.lin_in = nn.Linear(num_input, num_output)
        self.lin_out = nn.Linear(num_output, num_classes)
        # k = Phi.shape[1]
        self.filters = nn.Parameter(lamb_values.real)   #torch.randn(k) / (np.sqrt(num_input)))


    def forward(self, X):
        X = self.lin_in(X).float()
        X = torch.matmul(self.Phi.real, self.filters.reshape(-1,1) * torch.matmul(self.Phi.T.real, X))
        X = self.lin_out(X).float()
        return X
class GraphDMD(nn.Module):
    def __init__(self, num_features, num_latent, num_classes, Phi, lamb_values, dropout= 0.5):
        super().__init__()
        self.gcn1 = myGCNConv(num_features, num_latent, num_classes, Phi, lamb_values)
        self.dropout = dropout 
        self.m = nn.Dropout(p=self.dropout)
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_classes = num_classes




    def forward(self, X):
        X = self.gcn1(X)
        X = self.m(X)

        return F.log_softmax(X, dim=1)

# train/test    
def train(model, optimizer, data, train_mask, args):
    model.train()
    if args.model.lower() in ['gcn', 'gpr']:
        outputs = model(data.x, data.edge_index)
    elif args.model.lower() in ['glgcn']:
        outputs = model(data.x, data.y_mask, data.adj0)
    elif args.model.lower() in ['mlp']:
        outputs = model(data.x)
    elif args.model.lower() in ['appnp']:
        outputs = model(data.x, data.edge_index)
    elif args.model.lower() in ['h2']:
        outputs = model(data.x, data.adj_sparse)
    elif args.model.lower() in ['sage']:
        outputs = model(data.x, data.edge_index)
    elif args.model.lower() in ['graph_dmd']:
         outputs = model(data.x)    #, data.edge_index)


    loss = F.nll_loss(outputs[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    


def mmte(model, data, train_mask, val_mask, test_mask, args):
    model.eval()
    if args.model.lower() in ['gcn', 'gpr']:
        logits, accs = model(data.x, data.edge_index), []
    elif args.model.lower() in ['mlp']:
        logits, accs = model(data.x), []
    elif args.model.lower() in ['appnp']:
        logits, accs = model(data.x, data.edge_index), []
    elif args.model.lower() in ['h2']:
        logits, accs = model(data.x, data.adj_sparse), []
    elif args.model.lower() in ['sage']:
        logits, accs = model(data.x, data.edge_index), []
    if args.model.lower() in ['graph_dmd']:
        logits, accs = model(data.x), []


    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs



def load_data(args, root='data', rand_seed=2023):
    dataset = args.dataset
    path = osp.join(root, dataset)
    dataset = dataset.lower()


    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']


    except FileNotFoundError:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        elif dataset == 'cora_ml':
            dataset = CitationFull(path, dataset)
        elif dataset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dataset)
        elif dataset == 'actor':
            dataset = Actor(path)
        elif dataset in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(path, dataset)
        elif dataset in ['computers', 'photo']:
            dataset = Amazon(path, dataset)
        elif dataset in ['cs', 'physics']:
            dataset = Coauthor(path, dataset)


        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]


        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))


    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])


    return data, num_features, num_classes


def generate_split(data, num_classes, seed=2024, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask


    return train_mask.to(device), val_mask.to(device), test_mask.to(device)






# parameters 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, default='./',
                        help='Parent directory: please change to where you like')
    parser.add_argument("--directory", type=str, default='graph_storage',
                        help='Directory to store trained models')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='cora', help='Currently available: cora, citeseer, cornell, texas, wisconsin')
    parser.add_argument('--model', type=str, default='graph_dmd',
                        choices=['gcn', 'glgcn', 'mlp', 'graph_dmd'], help='GNN model')
    
    # optimization parameters 
    parser.add_argument('--runs', type=int, default=10, help='Number of repeating experiments for split.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.') 
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).') #0.0001
    parser.add_argument('--num_hid', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--train_rate',
                        type=float,
                        default=0.6,
                        help='Training rate.')  # used for heterophily for manually split. 
    parser.add_argument('--val_rate',
                        type=float,
                        default=0.2) # used for heterophily.




    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data, num_features, num_classes = load_data(args)
    
    
    results = []
    times = []

    adj_normalized = normalized_adjacency_matrix(data.edge_index, data.x.shape[0])
    X = normalize_features(data.x)
    data.x = torch.tensor(X)
    Y = adj_normalized.numpy()@ (adj_normalized.numpy() @ X)
    dmd = DMD(svd_rank=0.85, exact = True, opt=True) # # use extract = true for homo #svd_rank = 0.85 cora = 82.18
    dmd.fit(X, Y)
    print(len(dmd.eigs))
    Phi = torch.tensor(dmd.modes).float().to(device)#[:,idx[0]]
    lamb_values = torch.tensor(dmd.eigs).float().to(device)#[idx[0]]
    C = Phi.real.cpu() @ torch.diag(torch.tensor(dmd.eigs)) @ Phi.T.real.cpu()
    
    
    for run in range(args.runs):
          
        if args.dataset in ['texas', 'cornell', 'wisconsin', 'actor','chameleon','squirrel','photo','cs']:
            
            num_train = int(len(data.y) / num_classes*args.train_rate)
            num_val = int(len(data.y) / num_classes*args.val_rate)
            train_mask, val_mask, test_mask = generate_split(data, num_classes, args.seed, num_train, num_val)
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        
        
        
        log = 'Iteration {} starts.'
        print(log.format(run + 1))
        
        if args.model.lower() == 'gcn':
            model = GCN(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() == 'mlp':
            model = MLP(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() == 'appnp':
            model = APPNP(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() == 'gpr':
            model = GPRGNN(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() == 'h2':
            model = H2GCN(num_features, num_classes, args.num_hid, dropout=args.dropout) 
        elif args.model.lower() == 'sage':
            model = SAGE2(num_features, num_classes, args.num_hid,dropout=args.dropout)
        elif args.model.lower() == 'graph_dmd':    
            model = GraphDMD(num_features, args.num_hid, num_classes, Phi, lamb_values, dropout= args.dropout)


        model = model.to(device)
        data = data.to(device)


        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)


        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs + 1):


            train(model, optimizer, data, train_mask, args)
            with torch.no_grad():
                train_acc, val_acc, tmp_test_acc = mmte(model, data, train_mask, val_mask, test_mask, args)


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        t2 = time.time()
        results.append(test_acc)
        times.append(t2 - t1)


    results = 100 * torch.Tensor(results)
    times = torch.Tensor(times)
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f} in {times.mean():.2f} s')
    print(sum(times))
