import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from random_map import *

def linear_kernel(x, node_vec1, node_vec2):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3) # [N, B, 1, nhid]

    #node_vec1 = F.softmax(node_vec1, dim = 3 )
    
    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)

    v2x = F.softmax(v2x, dim = 3)

    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x) # [N, B, 1, nhid]
    
    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum) # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2 # [B, N, 1, nhid]

    return out1


def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    
    node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]] # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start) # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2) # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)
    
    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2) # [B, E, 1]
    attn_score = attn1 / attn2 # [B, E, 1]
    
    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    spatial_loss = torch.mean(attn_score.log() * d_norm)
    
    return spatial_loss

class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super(conv_approximation, self).__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(24, random_feature_dim, kernel_size=(1,1), bias=True)
        self.conv2 = nn.Conv2d(24, random_feature_dim, kernel_size=(1,1), bias=True)

    def forward(self, x, node_vec1, node_vec2):
        B = x.size(0) # (B, N, 1, nhid)
        dim = node_vec1.shape[-1] # (N, 1, d)
        
        x = linear_kernel(x, node_vec1, node_vec2)
        
        return x

class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super(linearized_conv, self).__init__()
        
        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        
        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)
        
    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
        x = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2) # (B, dim*4, N, 1)

        return x


########################


def linear_kernel1(x, source_emb, target_emb, node_vec1, node_vec2):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    #source[B,d,r,1]
    # x_k: [B, N, 1, nhid]
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3) # [N, B, 1, nhid]
    # x_k = x_k.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    source = source_emb.permute(0, 3, 1, 2) #[B, 1, r, d]
    target = target_emb
    #node_vec1 = F.softmax(node_vec1, dim = 3 )
    
    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x) #[B, 1, r, nhid]

    pool = []
    pool.append(v2x)
    pool.append(source)


    # print(v2x.shape)
    # print(source.shape)
    v2x = torch.cat(pool, dim=2)
    v2x = torch.einsum("nbri,rd->nbdi", v2x, target)


    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x) # [N, B, 1, nhid]
    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]

    return out1


class linearized_conv1(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super(linearized_conv1, self).__init__()
        
        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        
        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)
        
    def forward(self, input_data, source_emb, target_emb, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
        x = linear_kernel1(x, source_emb, target_emb, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2) # (B, dim*4, N, 1)

        return x

