
import os
import argparse
import math
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import scipy.sparse as sp
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score
import networkx as nx
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import edge_softmax
from models import HAMP as Model
# note: there may still be some variability
torch.manual_seed(8)
np.random.seed(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='screen_genre_class', type=str, help='selected task: screen_genre_class or element_comp_class')
parser.add_argument('--n_epochs', default=3000, type=int, help='number of epochs')
args = parser.parse_args()    

# paths
home_dir = Path(os.getcwd())
version_dir = 'rico_n'
main_data_dir = home_dir/'data'
data_dir = home_dir/'data'/version_dir


# load data - features
app2ui_edgelist = pd.read_hdf(data_dir/'app2ui_edgelist.h5', key='edgelist')
ui2class_edgelist = pd.read_hdf(data_dir/'ui2class_edgelist.h5', key='edgelist')
class2element_edgelist = pd.read_hdf(data_dir/'class2element_edgelist.h5', key='edgelist')
element2element_edgelist = pd.read_hdf(data_dir/'element2element_edgelist.h5', key='edgelist')
app_description_features_df = pd.read_hdf(data_dir/'app_description_features.h5', key='features')
ui_image_features = pd.read_hdf(data_dir/'ui_image_features.h5', key='features')
ui_pos_features_df = pd.read_hdf(data_dir/'ui_position_features.h5', key='features')
class_name_features = pd.read_hdf(data_dir/'charngram_features.h5', key='features') 
element_spatial_features_df = pd.read_hdf(data_dir/'spatial_features.h5', key='features')
element_image_features_df = pd.read_hdf(data_dir/'element_image_features.h5', key='features')
# load labels
comp_labels = pd.read_hdf(data_dir/'comp_labels.h5', key='labels')
genre_labels = pd.read_hdf(data_dir/'genre_labels.h5', key='labels')

# process edgelists
e2e_num_edges = len(element2element_edgelist)
e2e_adj_row = list(element2element_edgelist.target_element_encoded)
e2e_adj_col = list(element2element_edgelist.source_element_encoded)
e2e_num = len(element2element_edgelist.target_element_encoded.unique())
e2e_adj = sp.csc_matrix((np.ones(e2e_num_edges), (e2e_adj_row, e2e_adj_col)), shape=(e2e_num, e2e_num))
e2c_num_edges = len(class2element_edgelist)
e2c_adj_row = list(class2element_edgelist.target_element_encoded)
e2c_adj_col = list(class2element_edgelist.class_name_encoded)
e2c_num_row = len(class2element_edgelist.target_element_encoded.unique())
e2c_num_col = len(class2element_edgelist.class_name_encoded.unique())
e2c_adj = sp.csc_matrix((np.ones(e2c_num_edges), (e2c_adj_row, e2c_adj_col)), shape=(e2c_num_row, e2c_num_col))
u2c_num_edges = len(ui2class_edgelist)
u2c_adj_row = list(ui2class_edgelist.ui_encoded)
u2c_adj_col = list(ui2class_edgelist.class_name_encoded)
u2c_num_row = len(ui2class_edgelist.ui_encoded.unique())
u2c_num_col = len(ui2class_edgelist.class_name_encoded.unique())
u2c_adj = sp.csc_matrix((np.ones(u2c_num_edges), (u2c_adj_row, u2c_adj_col)), shape=(u2c_num_row, u2c_num_col))
a2u_num_edges = len(app2ui_edgelist)
a2u_adj_row = list(app2ui_edgelist.app_encoded)
a2u_adj_col = list(app2ui_edgelist.ui_encoded)
a2u_num_row = len(app2ui_edgelist.app_encoded.unique())
a2u_num_col = len(app2ui_edgelist.ui_encoded.unique())
a2u_adj = sp.csc_matrix((np.ones(a2u_num_edges), (a2u_adj_row, a2u_adj_col)), shape=(a2u_num_row, a2u_num_col))
# process features
assert (app_description_features_df.app_encoded == app2ui_edgelist.app_encoded.unique()).all()
app_desc_vectors = app_description_features_df.iloc[:,3:].values
assert (ui_image_features.ui_encoded == app2ui_edgelist.ui_encoded.unique()).all()
assert (ui_image_features.ui_encoded == ui2class_edgelist.ui_encoded.unique()).all()
ui_image_vectors = ui_image_features.iloc[:,2:].values
assert (class_name_features.class_name_encoded == ui2class_edgelist.class_name_encoded.unique()).all()
assert (class_name_features.class_name_encoded == ui2class_edgelist.class_name_encoded.unique()).all()
class_name_vectors = class_name_features.iloc[:,4:].values
assert (element_spatial_features_df.target_encoded == class2element_edgelist.target_element_encoded.unique()).all()
assert (element_spatial_features_df.target_encoded == element2element_edgelist.target_element_encoded.unique()).all()
element_spatial_vectors = element_spatial_features_df.iloc[:,2:].values
assert (element_image_features_df.target_element_encoded == class2element_edgelist.target_element_encoded.unique()).all()
assert (element_image_features_df.target_element_encoded == element2element_edgelist.target_element_encoded.unique()).all()
element_image_vectors = element_image_features_df.iloc[:,2:].values
assert (ui_image_features.ui_encoded == ui_pos_features_df.ui_encoded.unique()).all()
assert (ui_image_features.ui_encoded == ui_pos_features_df.ui_encoded.unique()).all()
ui_pos_vectors = ui_pos_features_df.iloc[:,3:].values

G = dgl.heterograph({
        ('element', 'fwd', 'element') : e2e_adj.nonzero(), # nonzero is the edgelist
        ('element', 'bkwd', 'element') : e2e_adj.transpose().nonzero(),
        ('element', 'is', 'class') : e2c_adj.nonzero(),
        ('class', 'of', 'element') : e2c_adj.transpose().nonzero(),
        # ('class', 'selfloop', 'class') : c2c_adj.nonzero(), # two-hop if necc
        ('ui', 'composed-of', 'class') : u2c_adj.nonzero(),
        ('class', 'in', 'ui') : u2c_adj.transpose().nonzero(),
        ('app', 'inc', 'ui') : a2u_adj.nonzero(),
        ('ui', 'part-of', 'app') : a2u_adj.transpose().nonzero(),
    }, num_nodes_dict = {'element': e2e_adj.shape[1], 'class': e2c_adj.shape[1], 'ui': a2u_adj.shape[1], 'app':a2u_adj.shape[0]})

# look-up dicts
node_dict = {}
reverse_node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    idx = len(node_dict)
    node_dict[ntype] = idx # increment by 1 each iteration
    reverse_node_dict[idx] = ntype
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    # assign a list of same integer ids to the etype
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] # attribute of edge with the id
    
node_feat = {'app':app_desc_vectors.shape[1], 'class':class_name_vectors.shape[1], 'element':element_image_vectors.shape[1], 'ui': ui_image_vectors.shape[1]}

G.nodes['element'].data['comp_label'] = torch.tensor(comp_labels.comp_encoded.values)
G.nodes['ui'].data['genre_label'] = torch.tensor(genre_labels.genre_encoded.values)
G.nodes['element'].data['node_ft'] = torch.tensor(element_image_vectors)
G.nodes['class'].data['node_ft'] = torch.tensor(class_name_vectors)
G.nodes['ui'].data['node_ft'] = torch.tensor(ui_image_vectors)
G.nodes['app'].data['node_ft'] = torch.tensor(app_desc_vectors)
element_spatial_vectors[:,4] = element_spatial_vectors[:,4] + 3
G.nodes['element'].data['pos'] = torch.FloatTensor(np.zeros((G.nodes('element').size()[0], 1)))
G.nodes['class'].data['pos'] = torch.FloatTensor(np.zeros((G.nodes('class').size()[0], 1)))
G.nodes['ui'].data['pos'] = torch.FloatTensor(ui_pos_vectors)
G.nodes['app'].data['pos'] = torch.FloatTensor(np.zeros((G.nodes('app').size()[0], 1)))
G.nodes['element'].data['depth'] = torch.FloatTensor(element_spatial_vectors[:,4]).unsqueeze(1)
G.nodes['class'].data['depth'] = torch.FloatTensor(np.ones((G.nodes('class').size()[0], 1))*3)
G.nodes['ui'].data['depth'] = torch.FloatTensor(np.ones((G.nodes('ui').size()[0], 1))*2)
G.nodes['app'].data['depth'] = torch.FloatTensor(np.ones((G.nodes('app').size()[0], 1))*1)
G.nodes['element'].data['bound1'] = torch.FloatTensor(element_spatial_vectors[:,0]).unsqueeze(1)
G.nodes['class'].data['bound1'] = torch.FloatTensor(np.zeros((G.nodes('class').size()[0], 1)))
G.nodes['ui'].data['bound1'] = torch.FloatTensor(np.zeros((G.nodes('ui').size()[0], 1)))
G.nodes['app'].data['bound1'] = torch.FloatTensor(np.zeros((G.nodes('app').size()[0], 1)))
G.nodes['element'].data['bound2'] = torch.FloatTensor(element_spatial_vectors[:,1]).unsqueeze(1)
G.nodes['class'].data['bound2'] = torch.FloatTensor(np.zeros((G.nodes('class').size()[0], 1)))
G.nodes['ui'].data['bound2'] = torch.FloatTensor(np.zeros((G.nodes('ui').size()[0], 1)))
G.nodes['app'].data['bound2'] = torch.FloatTensor(np.zeros((G.nodes('app').size()[0], 1)))
G.nodes['element'].data['bound3'] = torch.FloatTensor(element_spatial_vectors[:,2]).unsqueeze(1)
G.nodes['class'].data['bound3'] = torch.FloatTensor(np.zeros((G.nodes('class').size()[0], 1)))
G.nodes['ui'].data['bound3'] = torch.FloatTensor(np.zeros((G.nodes('ui').size()[0], 1)))
G.nodes['app'].data['bound3'] = torch.FloatTensor(np.zeros((G.nodes('app').size()[0], 1)))
G.nodes['element'].data['bound4'] = torch.FloatTensor(element_spatial_vectors[:,3]).unsqueeze(1)
G.nodes['class'].data['bound4'] = torch.FloatTensor(np.zeros((G.nodes('class').size()[0], 1)))
G.nodes['ui'].data['bound4'] = torch.FloatTensor(np.zeros((G.nodes('ui').size()[0], 1)))
G.nodes['app'].data['bound4'] = torch.FloatTensor(np.zeros((G.nodes('app').size()[0], 1)))

# initialize model
n_epoch = args.n_epochs
input_dim = None
hidden_dim = 64
clip = 1.0
max_lr=1e-3
task = args.task

if task == 'screen_genre_class':
    selected_element = 'ui'
    num_classes = len(G.nodes['ui'].data['genre_label'].unique())
    labels = G.nodes['ui'].data['genre_label']
    pid = u2c_adj.tocoo().row
elif task == 'element_comp_class':
    selected_element = 'element'
    num_classes = len(G.nodes['element'].data['comp_label'].unique())
    labels = G.nodes['element'].data['comp_label']
    pid = e2e_adj.tocoo().row

# generate train/val/test split
train = int(0.6*len(pid))
valid = int(0.8*len(pid))
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:train]).long()
val_idx = torch.tensor(shuffle[train:valid]).long()
test_idx = torch.tensor(shuffle[valid:]).long()

model = Model(node_dict, edge_dict, reverse_node_dict, node_feat, 
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=2,
            num_heads=2,
            use_layer_norm=True).to(device)

optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=n_epoch, max_lr = max_lr)
G = G.to(device) 

# train
best_val_acc = 0
best_test_acc = 0
best_micro_f1 = 0
best_macro_f1 = 0
train_step = 0
best_epoch = 0

for epoch in tqdm(np.arange(n_epoch)+1):
    model.train()
    logits = model(G, selected_element)
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    train_step += 1
    scheduler.step()
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(G, selected_element)
            pred   = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
            test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()

            test_micro_f1 = f1_score(labels[test_idx].detach().cpu().numpy(), pred[test_idx].numpy(), average='micro')
            test_macro_f1 = f1_score(labels[test_idx].detach().cpu().numpy(), pred[test_idx].numpy(), average='macro')

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                if (best_micro_f1 < test_micro_f1) & (best_macro_f1 < test_macro_f1):
                    best_micro_f1 = test_micro_f1
                    best_macro_f1 = test_macro_f1
                    best_epoch = epoch

print('='*100)
print(f'Test - Best - Micro F1: {best_micro_f1} | Macro F1: {best_macro_f1} | Best epoch: {best_epoch}')
print('='*100)




