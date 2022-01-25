import os
import math
import numpy as np
import pandas as pd
import json
import scipy.sparse as sp
from pathlib import Path
import networkx as nx
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import edge_softmax

class PosVect(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise=0.001):
        super().__init__()
        self.wave = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.softplus = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim*4, hidden_dim)

        # Initialize weights randomly with specific characteristics
        params = dict(self.named_parameters())
        params['wave.weight'].data = torch.from_numpy((2*np.pi*np.floor(np.arange(hidden_dim)/2))[:,np.newaxis]).float()
        params['wave.bias'].data = torch.from_numpy(np.pi/2+np.arange(hidden_dim)%2*np.pi/2).float()
        params['linear.weight'].data = torch.from_numpy(np.ones(shape=(hidden_dim,1)) + np.random.normal(size=(hidden_dim,1))*noise).float()
        params['linear.bias'].data = torch.from_numpy(np.random.normal(size=(hidden_dim))*noise).float()
        params['softplus.weight'].data = torch.from_numpy(np.random.normal(size=(hidden_dim,1))*noise).float()
        params['softplus.bias'].data = torch.from_numpy(np.random.normal(size=(hidden_dim))*noise).float()
        params['sigmoid.weight'].data = torch.from_numpy(np.random.normal(size=(hidden_dim,1))*noise).float()
        params['sigmoid.bias'].data = torch.from_numpy(np.random.normal(size=(hidden_dim))*noise).float()

    def forward(self, x):
        sinusoid = torch.sin(self.wave(x))
        linear = self.linear(x)
        softplus = nn.Softplus()(self.softplus(x))
        sigmoid = nn.Sigmoid()(self.sigmoid(x))
        combined = torch.cat([sinusoid, linear, softplus, sigmoid], dim=1)
        out = self.fc(combined)
        return out

class HAMPLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 node_dict, edge_dict,
                 num_heads, dropout = 0.2, use_layer_norm = True):
        super().__init__()
        
        self.input_dim = input_dim # input dimension after spatial-sequence-hierarichal vectorization, inter-modal and spatial-sequence-hierarichal attention and projection
        self.output_dim = output_dim # output dimension of each layer

        self.node_dict = node_dict # dictionaries contain the node-types and edge-tyeps
        self.edge_dict = edge_dict
        self.num_node_types = len(node_dict)
        self.num_edge_types = len(edge_dict)
        self.num_heads = num_heads 
        self.d_k = output_dim // num_heads 
        self.sqrt_dk = math.sqrt(self.d_k) 

        self.k_dense = nn.ModuleList()
        self.q_dense = nn.ModuleList()
        self.v_dense = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.use_layer_norm = use_layer_norm 
        self.canon_weights   = nn.Parameter(torch.ones(self.num_edge_types, self.num_heads))
        self.att_weights = nn.Parameter(torch.Tensor(self.num_edge_types, self.num_heads, self.d_k, self.d_k)) 
        self.value_weights = nn.Parameter(torch.Tensor(self.num_edge_types, self.num_heads, self.d_k, self.d_k)) 
        self.res = nn.Parameter(torch.ones(self.num_node_types))

        self.dropout = nn.Dropout(dropout)

        for t in range(self.num_node_types): 
            self.k_dense.append(nn.Linear(input_dim, output_dim))
            self.q_dense.append(nn.Linear(input_dim, output_dim))
            self.v_dense.append(nn.Linear(input_dim, output_dim))
            self.fc.append(nn.Linear(output_dim, output_dim))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(output_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_normal_(p, gain=0.001) # KIV use of nn.init.xavier_uniform_()
                nn.init.xavier_uniform_(p)

    def forward(self, G, h):
        with G.local_scope(): 
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_dense = self.k_dense[self.node_dict[srctype]] # this retrieves the models, node_dict number corresponds to dictionary position in the ModuleList
                v_dense = self.v_dense[self.node_dict[srctype]]
                q_dense = self.q_dense[self.node_dict[dsttype]]

                k = k_dense(h[srctype]).view(-1, self.num_heads, self.d_k) # source
                v = v_dense(h[srctype]).view(-1, self.num_heads, self.d_k) # source
                q = q_dense(h[dsttype]).view(-1, self.num_heads, self.d_k) # target

                # extract id for the edge
                e_id = self.edge_dict[etype]

                att_weights = self.att_weights[e_id] 
                canon_weights = self.canon_weights[e_id]
                value_weights = self.value_weights[e_id]

                k = torch.einsum("bij,ijk->bik", k, att_weights)
                v = torch.einsum("bij,ijk->bik", v, value_weights)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't')) 
                attn_score = sub_graph.edata.pop('t').sum(-1) * canon_weights / self.sqrt_dk 
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst') 

                sub_graph.edata['t'] = attn_score.unsqueeze(-1) 

            G.multi_update_all({etype : (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't')) for etype in self.edge_dict}, cross_reducer = 'mean')
            final_h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.res[n_id]) 
                t = G.nodes[ntype].data['t'].view(-1, self.output_dim) 
                h_prime = self.dropout(self.fc[n_id](t))
                h_prime = h_prime * alpha + h[ntype] * (1-alpha)
                if self.use_layer_norm:
                    final_h[ntype] = self.layer_norms[n_id](h_prime)
                else:
                    final_h[ntype] = h_prime
            return final_h 

class HAMP(nn.Module):
    def __init__(self, node_dict, edge_dict, reverse_node_dict, node_feat, input_dim, hidden_dim, output_dim, num_layers, num_heads, use_layer_norm = True, act = F.gelu):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.layer = nn.ModuleList()
        self.input_dim = input_dim # not used for now in HAMP as it is directly computed from node_feat dict, but can be utilized if all feature dimensions same
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act = act

        self.projection  = nn.ModuleList()

        self.pvect_pos  = nn.ModuleList()   
        self.pvect_depth  = nn.ModuleList()  
        self.pvect_bound1  = nn.ModuleList()  
        self.pvect_bound2  = nn.ModuleList()  
        self.pvect_bound3  = nn.ModuleList()
        self.pvect_bound4  = nn.ModuleList()  

        for t in range(len(node_dict)):
            self.pvect_pos.append(PosVect(1, hidden_dim))
            self.pvect_depth.append(PosVect(1, hidden_dim)) # depth corresponds to hierarchy information
            self.pvect_bound1.append(PosVect(1, hidden_dim))
            self.pvect_bound2.append(PosVect(1, hidden_dim))
            self.pvect_bound3.append(PosVect(1, hidden_dim))
            self.pvect_bound4.append(PosVect(1, hidden_dim))

            #project down from different feature dimensions
            in_dim = node_feat[reverse_node_dict[t]] # this is a lookup that allows us to assign different input dimension to each of the projection modules (depending on the modality of the feature)
            self.projection.append(nn.Linear(in_dim, hidden_dim))

        self.intermodal_attention = InterModalAttention(in_size=hidden_dim)

        for _ in range(num_layers):
            # input_dim here is hidden_dim as it has already been projected from the various node_feat to a common hidden_dim
            self.layer.append(HAMPLayer(hidden_dim, hidden_dim, node_dict, edge_dict, num_heads, use_layer_norm = use_layer_norm))
        
        self.out = nn.Linear(hidden_dim, output_dim) 
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_normal_(p, gain=0.001) # KIV use of nn.init.xavier_uniform_()
                nn.init.xavier_uniform_(p)

    def forward(self, G, sel_node_type):
        h = {}
        pos_h = {}
        depth_h = {}
        bound1_h = {}
        bound2_h = {}
        bound3_h = {}
        bound4_h = {}

        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = self.act(self.projection[n_id](G.nodes[ntype].data['node_ft'])) 

            pos_h[ntype] = self.act(self.pvect_pos[n_id](G.nodes[ntype].data['pos']))
            depth_h[ntype] = self.act(self.pvect_depth[n_id](G.nodes[ntype].data['depth']))
            bound1_h[ntype] = self.act(self.pvect_bound1[n_id](G.nodes[ntype].data['bound1']))
            bound2_h[ntype] = self.act(self.pvect_bound2[n_id](G.nodes[ntype].data['bound2']))
            bound3_h[ntype] = self.act(self.pvect_bound3[n_id](G.nodes[ntype].data['bound3']))
            bound4_h[ntype] = self.act(self.pvect_bound4[n_id](G.nodes[ntype].data['bound4']))

            # attention
            all_h = []
            all_h.append(h[ntype])
            all_h.append(pos_h[ntype])
            all_h.append(depth_h[ntype])
            all_h.append(bound1_h[ntype])
            all_h.append(bound2_h[ntype])
            all_h.append(bound3_h[ntype])
            all_h.append(bound4_h[ntype])

            all_h = torch.stack(all_h, dim=1) 
            att_h = self.intermodal_attention(all_h)  
            h[ntype] = att_h

        for i in range(self.num_layers):
            h = self.layer[i](G, h)

        return self.out(h[sel_node_type]) # select node type to get representation of

class InterModalAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)
