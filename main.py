import argparse
import os.path as osp
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool,GATConv,GCNConv,global_mean_pool,SAGEConv
from dataset import Route,Coordinate
from torch_geometric.datasets import TUDataset
import numpy as np
import time
from collections import  Counter
import math
time.clock = time.time
start = time.clock()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--heads', type=int, default=8)
args = parser.parse_args()

#dataset1 =TUDataset(root='./MYdata/processed', name='MUTAG')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GIN-{args.dataset}', batch_size=args.batch_size, lr=args.lr,
           epochs=args.epochs, hidden_channels=args.hidden_channels,
           num_layers=args.num_layers, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
#dataset = TUDataset(path, name=args.dataset).shuffle()
dataset = Route(root="./route/processed")
# dataset = Coordinate(root="./coordinate/processed")


# n = []
# degs = []
# for g in dataset:
#     num_nodes = g.num_nodes
#     deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
#     n.append(g.num_nodes)
#     degs.append(deg.max())
# print(f'Mean Degree: {torch.stack(degs).float().mean()}')
# print(f'Max Degree: {torch.stack(degs).max()}')
# print(f'Min Degree: {torch.stack(degs).min()}')
# mean_n = torch.tensor(n).float().mean().round().long().item()
# print(f'Mean number of nodes: {mean_n}')
# print(f'Max number of nodes: {torch.tensor(n).float().max().round().long().item()}')
# print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
# print(f'Number of graphs: {len(dataset)}')

train_dataset = dataset[len(dataset) // 10:]
train_bs = np.zeros(len(train_dataset),dtype=int)
num_nodes=int(dataset.data.num_nodes/len(dataset))
for i in range (0,len(train_dataset)):
    for j in range(0,num_nodes-1):
        if float(train_dataset[i].x[j][0])<-245:
            train_bs[i] = train_bs[i]+1

train_count = Counter(train_bs)
print("Training samples with", num_nodes, "bs:",train_count[0])
print("Training samples with", num_nodes-1, "bs:",train_count[1])
print("Training samples with", num_nodes-2, "bs:",train_count[2])
print("Training samples with", num_nodes-3, "bs:",train_count[3])
print("Training samples with", num_nodes-4, "bs:",train_count[4])
print("Training samples with", num_nodes-5, "bs:",train_count[5])

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

test_dataset = dataset[:len(dataset) // 10]
test_bs = np.zeros(len(test_dataset),dtype=int)
for i in range (0,len(test_dataset)):
    for j in range(0,num_nodes-1):
        if float(test_dataset[i].x[j][0])<-245:
            test_bs[i] = test_bs[i]+1
test_count = Counter(test_bs)
print("Testing samples with", num_nodes,"bs:",test_count[0])
print("Testing samples with", num_nodes-1,"bs:",test_count[1])
print("Testing samples with", num_nodes-2,"bs:",test_count[2])
print("Testing samples with", num_nodes-3,"bs:",test_count[3])
print("Testing samples with", num_nodes-4,"bs:",test_count[4])
print("Testing samples with", num_nodes-5,"bs:",test_count[5])

test_loader = DataLoader(test_dataset, args.batch_size)

class GNNnet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNnet, self).__init__()
        self.lin_1 = MLP([num_node_features,  32, 64])
        self.lin_2 = MLP([64, 32, num_classes])
        self.gat1 = GATConv(64, 64, heads=1, cancat=True,dropout=0.6,add_self_loops = True)
        self.gat2= GATConv(64, 64, heads=1, concat=False, dropout=0.6, add_self_loops=True)
        self.gcn1 = GCNConv(64, 128,cached=False,add_self_loops = True,normalize=True)
        self.gcn2 = GCNConv(128, 64,cached=False, add_self_loops=True, normalize=True)
        mlp1 = MLP([64, 128])
        mlp2 = MLP([128, 64])
        self.gin1 = GINConv(nn=mlp1, train_eps=True)
        self.gin2 = GINConv(nn=mlp2, train_eps=True)
        self.lin_11 = MLP([64, 128])
        self.lin_22 = MLP([128, 64])
        self.sage1 = SAGEConv(64,128)
        self.sage2 = SAGEConv(128, 64)

    def forward(self, x, edge_index, edge_weight, batch):

        x = self.lin_1(x)
        x = F.leaky_relu(x)

        #Feature Extraction
        #Use different GNNs or MLP
        ##########################
        x = self.sage1(x, edge_index)
        # x = self.lin_11(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        # x = self.lin_22(x)
        x = F.leaky_relu(x)
        ###########################

        #Linear layer and node_level_pooling, to get a graph level output
        x = self.lin_2(x)
        x = global_add_pool(x,batch)

        return x


model = GNNnet(dataset.num_features,dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination

    latitude1 = (math.pi/180)*lat1
    latitude2 = (math.pi/180)*lat2
    longitude1 = (math.pi/180)*lon1
    longitude2= (math.pi/180)*lon2

    R = 6378.137
    d = math.acos(math.sin(latitude1)*math.sin(latitude2)+ math.cos(latitude1)*math.cos(latitude2)*math.cos(longitude2-longitude1))*R
    return d

#####################################################
#Classification
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.y=data.y.type(torch.LongTensor)
        data.y=data.y.to(device)
        out=model(data.x,data.edge_index,data.edge_weight,data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x,  data.edge_index, data.edge_weight, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
###################################################################################################
###################################################################################################
# #Regression
# def train():
#     model.train()
#
#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         data.y=data.y.to(device)
#
#         #For regression, the output of model should be x instead of softmax(x)
#         out=model(data.x,data.edge_index,data.edge_weight,data.batch)
#
#         #Use distance as loss
#         # dist=np.zeros(len(out))
#         # for i in range(0, len(out)):
#         #     dist[i] = distance(out[i], dataset.y[i])
#         # loss = sum(a ** 2 for a in dist) / len(dist)
#
#         loss = F.mse_loss(out, data.y)
#
#         #if use distance as loss, need to transfer array to learnable variable
#         # loss = torch.from_numpy(np.array([loss])).to(torch.float32)
#         # loss = Variable(loss, requires_grad=True)
#
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * data.num_graphs
#
#         #if use distance as loss
#         # total_loss += float(loss) * len(dist)
#
#     return math.sqrt(total_loss / len(train_loader.dataset))
#
# def test(loader):
#     model.eval()
#     test_loss = 0
#     for data in loader:
#         data = data.to(device)
#         pred = model(data.x, data.edge_index, data.edge_weight,data.batch)
#
#         #Use distance as loss
#         # dist=np.zeros(len(pred))
#         # for i in range(1, len(pred)):
#         #     dist[i] = distance(pred[i], dataset.y[i])
#         # loss = sum(a ** 2 for a in dist) / len(dist)
#
#         loss = F.mse_loss(pred, data.y)
#         test_loss += float(loss) * data.num_graphs
#
#         #if use distance as loss
#         # test_loss += float(loss) * len(dist)
#
#     return math.sqrt(test_loss / len(loader.dataset))
#
# for epoch in range(1, args.epochs + 1):
#     loss = train()
#     test_loss = test(test_loader)
#     log(Epoch=epoch, Train=loss,  Test=test_loss)

end = time.clock()
print("Time", end-start)