import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_conv import *
from torch.autograd import Variable


class GSNet(nn.Module):
    def __init__(self, device, seq_num, in_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                 use_residual, use_bn, use_spatial, use_long, dropout, supports=None, edge_indices=None):
        super(GSNet, self).__init__()
        self.tau = tau
        self.layer_num = 3
        self.layer_num = 1
        self.random_feature_dim = random_feature_dim
        
        self.use_residual = use_residual
        #self.use_residual = False
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.use_long = use_long
        
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.supports = supports
        self.edge_indices = edge_indices
        
        self.time_num = 288
        self.week_num = 7
        
        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        self.source_emb_layer = nn.Parameter(torch.empty(hid_dim*2, hid_dim))
        nn.init.xavier_uniform_(self.source_emb_layer)
        self.target_emb_layer = nn.Parameter(torch.empty(hid_dim*2, hid_dim*2))
        nn.init.xavier_uniform_(self.target_emb_layer)
        
        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(seq_num, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.W_1 = nn.Conv2d(node_emb_dim, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim, hid_dim, kernel_size=(1, 1), bias=True)

        self.W_3 = nn.Conv2d(node_emb_dim * 2, hid_dim*2, kernel_size=(1, 1), bias=True)
        self.W_4 = nn.Conv2d(node_emb_dim * 2, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.linear_conv = nn.ModuleList()
        self.linear_conv1 = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.Klinear = nn.ModuleList()
        self.bn_k = nn.ModuleList()

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        for i in range(self.layer_num):
            self.linear_conv.append(linearized_conv(hid_dim*2, hid_dim*2, self.dropout, self.tau, self.random_feature_dim))
            self.linear_conv1.append(linearized_conv1(hid_dim*2, hid_dim*2, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim*2))
            self.Klinear.append(nn.Conv2d(hid_dim*2, hid_dim*2, kernel_size=(1, 1), bias=True))
            self.bn_k.append(nn.LayerNorm(hid_dim*2))
        
        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim*4*2+hid_dim+seq_num, 12, kernel_size=(1, 1), bias=True)
        else:
            # self.regression_layer = nn.Conv2d(hid_dim*8, 12, kernel_size=(1, 1), bias=True)
            #self.regression_layer = nn.Conv2d(hid_dim*6, 12, kernel_size=(1, 1), bias=True)
            self.regression_layer = nn.Conv2d(hid_dim*4, 12, kernel_size=(1, 1), bias=True)


    def forward(self, x, feat=None):
        # input: (B, N, T, D)
        B, N, T, D = x.size()
        x = x[..., 0]
        
        #time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
        #week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]
        
        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)

        source_emb = self.source_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, dim, 1)
        target_emb = self.target_emb_layer

        # time embeddings
        #time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        #week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        
        x_g = torch.cat([node_emb], dim=1) # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb], dim=1) # (B, dim*4, N, 1)
        x_e = torch.cat([input_emb, node_emb], dim=1)

        # linearized spatial convolution
        x_pool = [x] # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g) # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g) # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1) # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1) # (B, N, 1, dim)

        node_vec3 = self.W_3(x_e).permute(0, 2, 3, 1) # (B, dim, N, 1)
        node_vec4 = self.W_4(x_e).permute(0, 2, 3, 1) # (B, dim, N, 1)

        x_k = x
        for i in range(self.layer_num):
            if self.use_residual:
                residual = x
                residual_k = x_k
            x = self.linear_conv[i](x, node_vec1, node_vec2)

            x_k = self.linear_conv1[i](x_k, source_emb, target_emb, node_vec3, node_vec4)
            #x = self.Klinear[i](x)
            
            if self.use_residual:
                x = x+residual 
                x_k = x_k+residual 
                
            if self.use_bn:
                x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

                x_k = x_k.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
                x_k = self.bn_k[i](x_k)
                x_k = x_k.permute(0, 3, 1, 2)

        x_pool.append(x)
        #x_pool.append(x_k)
        x = torch.cat(x_pool, dim=1) # (B, dim*4, N, 1)
        
        x = self.activation(x) # (B, dim*4, N, 1)
        
        # if self.use_long:
        #     feat = feat.permute(0, 2, 1).unsqueeze(-1) # (B, F, N, 1)
        #     x = torch.cat([x, feat], dim=1)
        #     x = self.regression_layer(x) # (B, N, T)
        #     x = x.squeeze(-1).permute(0, 2, 1)
        # else:
        x = self.regression_layer(x) # (B, N, T)
            # x = self.regression_layer(F.relu(x)) # (B, N, T)
        x = x.squeeze(-1).permute(0, 2, 1)

            
        
        # if self.use_spatial:
        #     s_loss = spatial_loss(node_vec1_prime, node_vec2_prime, self.supports, self.edge_indices)
        #     return x, s_loss
        # else:
        return x, 0

