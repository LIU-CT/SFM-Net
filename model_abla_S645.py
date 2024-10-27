# -*- coding: utf-8 -*-
"""
@author: liu
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import math
import numpy as np

from timm.layers import DropPath, to_2tuple, trunc_normal_, _assert
#from torch_geometric.nn import GATConv as GCNConv

from torch_geometric.nn import GINConv as GCNConv



#######
class silu(torch.nn.Module):
    def __init__(self):
        super(silu, self).__init__()

    @torch.jit.script
    def compute(x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        return self.compute(x)
    
  
##############################


class mlp(torch.nn.Module):
    def __init__(self, dim, dropout):
        super(mlp, self).__init__()
        self.proj1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj2(x)
        
        return x

class GSABlock(torch.nn.Module):
    def __init__(self, dim, node, dropout):
        super(GSABlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.qkv = nn.Linear(dim, dim * 3)
        self.ffn = mlp(dim, dropout)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.g = np.max([dim // 32, 1])
        self.d = dim//self.g
        self.scale = 1./math.sqrt(self.d)
        
        self.node = node # node = 10
        
    def forward(self, x, site):
        b, c = x.shape
        g, d = self.g, self.d
        
        x = self.norm1(x)
        q, k, v = self.qkv(x).view(1, b, 3, g, d).transpose(1, 3).chunk(3, dim=2) # 1 g b d
        
        sim = self.scale * q @ k.transpose(-2, -1)
        sim = self.softmax(sim)
        attn = sim @ v # 1 g b d
        attn = attn.transpose(1, 2).reshape(b, c)
        
        attn = self.proj(attn)
        attn = self.dropout(attn)
        
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x

###############################################################################
### vx verx
class GSABlockvz3(torch.nn.Module):
    def __init__(self, dim, node, dropout):
        super(GSABlockvz3, self).__init__()
        self.nx1 = nn.LayerNorm(dim, eps=1e-06)
        self.nx2 = nn.LayerNorm(dim, eps=1e-06)
        self.nx3 = nn.LayerNorm(dim, eps=1e-06)
        self.nx4 = nn.LayerNorm(dim, eps=1e-06)
        self.nx5 = nn.LayerNorm(dim, eps=1e-06)
        self.nx6 = nn.LayerNorm(dim, eps=1e-06)
        self.nx7 = nn.LayerNorm(dim, eps=1e-06)
        self.nx8 = nn.LayerNorm(dim, eps=1e-06)
        self.nx9 = nn.LayerNorm(dim, eps=1e-06)
        self.nxq = nn.LayerNorm(dim, eps=1e-06)

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.ffn = mlp(dim, dropout)

        self.proj = nn.Linear(dim * 7, dim)
        #self.proj = nn.Linear(dim, dim)
        self.postq = nn.Linear(dim * 9, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

        self.softmax = nn.Softmax(dim=-1)

        self.g = np.max([dim // 32, 1])
        self.d = dim//self.g
        self.scale = 1./math.sqrt(self.d)

        self.node = node # node = 10

    def forward(self, x, site):
        b, c = x.shape
        g, d = self.g, self.d

        n = b//9

        x = x.view(9, n, c)
        q_ = x[:6, :, :].mean(dim=0)

        x1 = self.nx1(x[0])
        x2 = self.nx2(x[1])
        x3 = self.nx3(x[2])
        x4 = self.nx4(x[3])
        x5 = self.nx5(x[4])
        x6 = self.nx6(x[5])

        x7 = self.nx7(x[6])
        x8 = self.nx8(x[7])
        x9 = self.nx9(x[8])
        qn = self.nxq(q_)

        x = torch.cat([qn, x1, x2, x3, x4, x5, x6, x7, x8, x9], dim=0).view(10, n, c)

        q = x[:7, :, :] # 6 N c
        q = self.q(q).view(7, n, g, d).permute(1, 2, 0, 3) # n g 3 d
        k = self.k(x).view(10, n, g, d).permute(1, 2, 0, 3) # n g 9 d
        v = self.v(x).view(10, n, g, d).permute(1, 2, 0, 3) # n g 9 d

        sim = self.scale * q @ k.transpose(-2, -1) # 2 6
        sim = self.softmax(sim) # 2 6
        attn = sim @ v # 2

        attn = attn.transpose(1, 2).reshape(n, 7*c)
        #attn = attn.transpose(1, 2).reshape(n, c)

        attn = self.proj(attn)
        attn = self.dropout(attn)

        #q = q.transpose(1, 2).reshape(n, 2*c)
        #q = self.postq(q)
        q = q_
        #x = x.transpose(0, 1).reshape(n, 6*c)
        #q = self.postq(x)
        #q = self.dropout(q)

        x = q + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x
    
    
    
#########
# SFM_Net
class SFM_Net(torch.nn.Module):
    def __init__(self):
        super(SFM_Net, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645: SFM_Net, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3(1024, 10, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)#0801版本将dropout固定为0.1 OR 0.2 or0.3 or0
        self.gsa3 = GSABlockvz3(1024, 10, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        ##
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)

        ##
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)

        ##
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)

        ##
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)


        x = torch.cat([self.proj1(x_wt1), self.proj1(x_mt)], dim=1)

        residual = self.proj2(x_mt - x_wt1)

        residual = self.res2(residual)

        x = self.mlp(x)

        x = self.drop_path(self.norm2(residual)) + self.norm1(x)

        x = F.silu(x)

        x = self.dropout_heavy(x)

        x = self.proj3(x)



        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec

        residual = x * cindexvec.view(-1, 1) # sparse vector of 0,1
        residual = pyg_nn.global_add_pool(residual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)


        # mean pool
        avgx = pyg_nn.global_mean_pool(x, batch_mt)



        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()

        x = x * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        x = pyg_nn.global_add_pool(x, batch_mt) * ratio_vec.view(-1, 1)

        #x = self.gsa2(x, edge_index_mt)


        b, c = x.shape



        y = self.proj1dssp(x_wt2)

        # center-pool
        yresidual = y * cindexvec.view(-1, 1) # sparse vector of 0,1
        yresidual = pyg_nn.global_add_pool(yresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgy = pyg_nn.global_mean_pool(y, batch_mt)


        # relevant-pool v2 (even summation)
        y = y * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        y = pyg_nn.global_add_pool(y, batch_mt) * ratio_vec.view(-1, 1)


        z = self.proj1pssm(x_wt3)

        # center-pool
        zresidual = z * cindexvec.view(-1, 1) # sparse vector of 0,1
        zresidual = pyg_nn.global_add_pool(zresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgz = pyg_nn.global_mean_pool(z, batch_mt)

        # relevant-pool v2 (even summation)
        z = z * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        z = pyg_nn.global_add_pool(z, batch_mt) * ratio_vec.view(-1, 1)

        x = torch.cat([self.cbn1(x), self.cbn2(y), self.cbn3(z),
                       self.cbn4(residual), self.cbn5(yresidual), self.cbn6(zresidual), avgx, avgy, avgz], dim=0)

        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)

        x = self.norm5(self.gsa2(x, edge_index_mt))


        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x





################## ablation study 01 ######################
# MLP--Diff component & attention structure
###########################################################
class SFM_Netablastu01(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu01, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8#0.3
        
        print(f'S645: SFM_Netablastu01, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3(1024, 10, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3(1024, 10, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)

        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        ##
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)

        ##
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)

        ##
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)

        ##
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)

        # mean pool
        avgx_wt1 = pyg_nn.global_mean_pool(x_wt1, batch_wt)
        # mean pool
        avgx_mt = pyg_nn.global_mean_pool(x_mt, batch_mt)
        # mean pool
        avgx_wt2 = pyg_nn.global_mean_pool(x_wt2, batch_wt)
        # mean pool
        avgx_wt3 = pyg_nn.global_mean_pool(x_wt3, batch_wt)        
        #print(avgx_wt1.shape, avgx_mt.shape, avgx_wt2.shape, avgx_wt3.shape)
        
        x = torch.cat([self.cbn1(avgx_wt1), self.cbn2(avgx_mt), self.cbn3(avgx_wt2),
                       self.cbn4(avgx_wt3)], dim=0)
        #print(x.shape)
        b, c = avgx_wt1.shape
        x = x.view(4, b, c)
        #print(x.shape)
        x = torch.mean(x, dim=0, keepdim=True)
        #print(x.shape)
        x = self.mlp2(x)
        #print(x.shape)

        self.v0 = x.shape
        x = x.squeeze()

        return x

    

#############################################################

################## ablation study 02 #################
# SFM_Netablastu02 prot5 without dssp&pssm 
# SFM_Netablastu02v1 dssp without prot5&pssm
# SFM_Netablastu02v2 pssm without prot5&dssp
######################################################
# SFM_Netablastu02 SFM_Netablastu02v1 SFM_Netablastu02v2: GSABlockvz3ablastu02
######################################################
class GSABlockvz3ablastu02(torch.nn.Module):
    def __init__(self, dim, node, dropout):
        super(GSABlockvz3ablastu02, self).__init__()
        self.nx1 = nn.LayerNorm(dim, eps=1e-06)
        self.nx2 = nn.LayerNorm(dim, eps=1e-06)
        self.nx3 = nn.LayerNorm(dim, eps=1e-06)
        self.nx4 = nn.LayerNorm(dim, eps=1e-06)
        self.nx5 = nn.LayerNorm(dim, eps=1e-06)
        self.nx6 = nn.LayerNorm(dim, eps=1e-06)
        self.nx7 = nn.LayerNorm(dim, eps=1e-06)
        self.nx8 = nn.LayerNorm(dim, eps=1e-06)
        self.nx9 = nn.LayerNorm(dim, eps=1e-06)
        self.nxq = nn.LayerNorm(dim, eps=1e-06)

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.ffn = mlp(dim, dropout)

        self.proj = nn.Linear(dim * 3, dim)
        #self.proj = nn.Linear(dim, dim)
        self.postq = nn.Linear(dim * 9, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

        self.softmax = nn.Softmax(dim=-1)

        self.g = np.max([dim // 32, 1])
        self.d = dim//self.g
        self.scale = 1./math.sqrt(self.d)

        self.node = node # node = 10

    def forward(self, x, site):
        b, c = x.shape
        #print(b,c)
        g, d = self.g, self.d

        n = b//3

        x = x.view(3, n, c)
        q_ = x[:2, :, :].mean(dim=0)

        x1 = self.nx1(x[0])
        x2 = self.nx2(x[1])
        x3 = self.nx3(x[2])

        qn = self.nxq(q_)

        x = torch.cat([qn, x1, x2, x3], dim=0).view(4, n, c)

        q = x[:3, :, :] # 6 N c
        q = self.q(q).view(3, n, g, d).permute(1, 2, 0, 3) # n g 3 d
        k = self.k(x).view(4, n, g, d).permute(1, 2, 0, 3) # n g 9 d
        v = self.v(x).view(4, n, g, d).permute(1, 2, 0, 3) # n g 9 d

        sim = self.scale * q @ k.transpose(-2, -1) # 2 6
        sim = self.softmax(sim) # 2 6
        attn = sim @ v # 2

        attn = attn.transpose(1, 2).reshape(n, 3*c)
        #attn = attn.transpose(1, 2).reshape(n, c)

        attn = self.proj(attn)
        attn = self.dropout(attn)

        #q = q.transpose(1, 2).reshape(n, 2*c)
        #q = self.postq(q)
        q = q_
        #x = x.transpose(0, 1).reshape(n, 6*c)
        #q = self.postq(x)
        #q = self.dropout(q)

        x = q + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


################## ablation study 02 #################
# SFM_Netablastu02 prot5 without dssp&pssm 
######################################################
class SFM_Netablastu02(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu02, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645: SFM_Netablastu02, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        ##
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)

 
        ##
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)


        x = torch.cat([self.proj1(x_wt1), self.proj1(x_mt)], dim=1)

        residual = self.proj2(x_mt - x_wt1)

        residual = self.res2(residual)

        x = self.mlp(x)

        x = self.drop_path(self.norm2(residual)) + self.norm1(x)

        x = F.silu(x)

        x = self.dropout_heavy(x)

        x = self.proj3(x)



        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec

        residual = x * cindexvec.view(-1, 1) # sparse vector of 0,1
        residual = pyg_nn.global_add_pool(residual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)


        # mean pool
        avgx = pyg_nn.global_mean_pool(x, batch_mt)



        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()

        x = x * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        x = pyg_nn.global_add_pool(x, batch_mt) * ratio_vec.view(-1, 1)



        b, c = x.shape
        x = torch.cat([self.cbn1(x), self.cbn4(residual), avgx], dim=0)
        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x


################## ablation study 02 #################
# SFM_Netablastu02v1 dssp without prot5&pssm
######################################################
class SFM_Netablastu02v1(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu02v1, self).__init__()

        dropout = 0.8 #0.1

        dropout_heavy = 0.8 #0.3
        
        print(f'S645: SFM_Netablastu02v1, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)

        ## DSSP GNN
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)


        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec 
        
        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()
        
        # dssp
        y = self.proj1dssp(x_wt2)

        # center-pool
        yresidual = y * cindexvec.view(-1, 1) # sparse vector of 0,1
        yresidual = pyg_nn.global_add_pool(yresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgy = pyg_nn.global_mean_pool(y, batch_mt)

        # relevant-pool v2 (even summation)
        y = y * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        y = pyg_nn.global_add_pool(y, batch_mt) * ratio_vec.view(-1, 1)        


        b, c = y.shape
        x = torch.cat([self.cbn1(y), self.cbn4(yresidual), avgy], dim=0)
        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x




################## ablation study 02 #################
# SFM_Netablastu02v2 pssm without prot5&dssp
######################################################
class SFM_Netablastu02v2(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu02v2, self).__init__()

        dropout = 0.8 #0.1

        dropout_heavy = 0.8 #0.3
        
        print(f'S645: SFM_Netablastu02v2, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu02(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        # PSSM GNN
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)
        

        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec 
        
        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()
        
        # pssm
        z = self.proj1pssm(x_wt3)

        # center-pool
        zresidual = z * cindexvec.view(-1, 1) # sparse vector of 0,1
        zresidual = pyg_nn.global_add_pool(zresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgz = pyg_nn.global_mean_pool(z, batch_mt)

        # relevant-pool v2 (even summation)
        z = z * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        z = pyg_nn.global_add_pool(z, batch_mt) * ratio_vec.view(-1, 1)     


        b, c = z.shape
        x = torch.cat([self.cbn1(z), self.cbn4(zresidual), avgz], dim=0)
        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x


#########################################################################
################## ablation study 03 #################
# SFM_Netablastu03   dssp&pssm   without prot5
# SFM_Netablastu03v1 prot5&pssm  without dssp 
# SFM_Netablastu03v2 prot5&dssp  without pssm 
######################################################
# SFM_Netablastu03 SFM_Netablastu03v1 SFM_Netablastu03v: GSABlockvz3ablastu03
######################################################

class GSABlockvz3ablastu03(torch.nn.Module):
    def __init__(self, dim, node, dropout):
        super(GSABlockvz3ablastu03, self).__init__()
        self.nx1 = nn.LayerNorm(dim, eps=1e-06)
        self.nx2 = nn.LayerNorm(dim, eps=1e-06)
        self.nx3 = nn.LayerNorm(dim, eps=1e-06)
        self.nx4 = nn.LayerNorm(dim, eps=1e-06)
        self.nx5 = nn.LayerNorm(dim, eps=1e-06)
        self.nx6 = nn.LayerNorm(dim, eps=1e-06)
        self.nx7 = nn.LayerNorm(dim, eps=1e-06)
        self.nx8 = nn.LayerNorm(dim, eps=1e-06)
        self.nx9 = nn.LayerNorm(dim, eps=1e-06)
        self.nxq = nn.LayerNorm(dim, eps=1e-06)

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.ffn = mlp(dim, dropout)

        self.proj = nn.Linear(dim * 5, dim)
        #self.proj = nn.Linear(dim, dim)
        self.postq = nn.Linear(dim * 9, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

        self.softmax = nn.Softmax(dim=-1)

        self.g = np.max([dim // 32, 1])
        self.d = dim//self.g
        self.scale = 1./math.sqrt(self.d)

        self.node = node # node = 10

    def forward(self, x, site):
        b, c = x.shape
        #print(b,c)
        g, d = self.g, self.d

        n = b//6

        x = x.view(6, n, c)
        q_ = x[:4, :, :].mean(dim=0)

        x1 = self.nx1(x[0])
        x2 = self.nx2(x[1])
        x3 = self.nx3(x[2])
        x4 = self.nx4(x[3])
        x5 = self.nx5(x[4])
        x6 = self.nx6(x[5])

        qn = self.nxq(q_)

        x = torch.cat([qn, x1, x2, x3, x4, x5, x6], dim=0).view(7, n, c)

        q = x[:5, :, :] # 6 N c
        q = self.q(q).view(5, n, g, d).permute(1, 2, 0, 3) # n g 3 d
        k = self.k(x).view(7, n, g, d).permute(1, 2, 0, 3) # n g 9 d
        v = self.v(x).view(7, n, g, d).permute(1, 2, 0, 3) # n g 9 d

        sim = self.scale * q @ k.transpose(-2, -1) # 2 6
        sim = self.softmax(sim) # 2 6
        attn = sim @ v # 2

        attn = attn.transpose(1, 2).reshape(n, 5*c)
        #attn = attn.transpose(1, 2).reshape(n, c)

        attn = self.proj(attn)
        attn = self.dropout(attn)

        #q = q.transpose(1, 2).reshape(n, 2*c)
        #q = self.postq(q)
        q = q_
        #x = x.transpose(0, 1).reshape(n, 6*c)
        #q = self.postq(x)
        #q = self.dropout(q)

        x = q + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x
    
    
################## ablation study 03 #################
# SFM_Netablastu03 dssp&pssm  without prot5
######################################################
class SFM_Netablastu03(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu03, self).__init__()

        dropout = 0.8 #0.1

        dropout_heavy = 0.8 #0.3
        
        print(f'SFM_Netablastu03, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        # DSSP GNN
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)

        # PSSM GNN
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)



        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec 
        
        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()
        
        # dssp
        y = self.proj1dssp(x_wt2)

        # center-pool
        yresidual = y * cindexvec.view(-1, 1) # sparse vector of 0,1
        yresidual = pyg_nn.global_add_pool(yresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgy = pyg_nn.global_mean_pool(y, batch_mt)

        # relevant-pool v2 (even summation)
        y = y * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        y = pyg_nn.global_add_pool(y, batch_mt) * ratio_vec.view(-1, 1)  
        
        # pssm
        z = self.proj1pssm(x_wt3)

        # center-pool
        zresidual = z * cindexvec.view(-1, 1) # sparse vector of 0,1
        zresidual = pyg_nn.global_add_pool(zresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgz = pyg_nn.global_mean_pool(z, batch_mt)

        # relevant-pool v2 (even summation)
        z = z * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        z = pyg_nn.global_add_pool(z, batch_mt) * ratio_vec.view(-1, 1)  



        b, c = z.shape
        x = torch.cat([self.cbn2(y), self.cbn3(z),
                       self.cbn5(yresidual), self.cbn6(zresidual), 
                       avgy, avgz], dim=0)

        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x


################## ablation study 03 #################
# SFM_Netablastu03v1 prot5&pssm  without dssp 
######################################################
class SFM_Netablastu03v1(torch.nn.Module): 
    def __init__(self):
        super(SFM_Netablastu03v1, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645: SFM_Netablastu03v1, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)

        # Prot5WT GNN
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)
        
        # Prot5MT GNN
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)

        # WT&MT SUM&DIFF 
        x = torch.cat([self.proj1(x_wt1), self.proj1(x_mt)], dim=1)

        residual = self.proj2(x_mt - x_wt1)

        residual = self.res2(residual)

        x = self.mlp(x)

        x = self.drop_path(self.norm2(residual)) + self.norm1(x)

        x = F.silu(x)

        x = self.dropout_heavy(x)

        x = self.proj3(x)        



        # PSSM GNN
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)



        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec 
        
        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()
        
        # WT&MT
        # center-pool
        residual = x * cindexvec.view(-1, 1) # sparse vector of 0,1
        residual = pyg_nn.global_add_pool(residual, batch_mt)
        
        # mean pool
        avgx = pyg_nn.global_mean_pool(x, batch_mt)

        # relevant-pool v2 (even summation)
        x = x * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        x = pyg_nn.global_add_pool(x, batch_mt) * ratio_vec.view(-1, 1)        
        

        
        # pssm
        z = self.proj1pssm(x_wt3)

        # center-pool
        zresidual = z * cindexvec.view(-1, 1) # sparse vector of 0,1
        zresidual = pyg_nn.global_add_pool(zresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgz = pyg_nn.global_mean_pool(z, batch_mt)

        # relevant-pool v2 (even summation)
        z = z * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        z = pyg_nn.global_add_pool(z, batch_mt) * ratio_vec.view(-1, 1)  



        b, c = z.shape
        x = torch.cat([self.cbn1(x), self.cbn3(z),
                       self.cbn4(residual), self.cbn6(zresidual), avgx, avgz], dim=0)


        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x





################## ablation study 03 #################
# SFM_Netablastu03v2 prot5&dssp  without pssm 
######################################################
class SFM_Netablastu03v2(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu03v2, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645: SFM_Netablastu03v2, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)
        self.gsa3 = GSABlockvz3ablastu03(1024, 3, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)

        # Prot5WT GNN
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)
        
        # Prot5MT GNN
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)

        # WT&MT SUM&DIFF 
        x = torch.cat([self.proj1(x_wt1), self.proj1(x_mt)], dim=1)

        residual = self.proj2(x_mt - x_wt1)

        residual = self.res2(residual)

        x = self.mlp(x)

        x = self.drop_path(self.norm2(residual)) + self.norm1(x)

        x = F.silu(x)

        x = self.dropout_heavy(x)

        x = self.proj3(x)        



        # DSSP GNN
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)


        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec 
        
        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()


        # WT&MT
        # center-pool
        residual = x * cindexvec.view(-1, 1) # sparse vector of 0,1
        residual = pyg_nn.global_add_pool(residual, batch_mt)
        
        # mean pool
        avgx = pyg_nn.global_mean_pool(x, batch_mt)

        # relevant-pool v2 (even summation)
        x = x * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        x = pyg_nn.global_add_pool(x, batch_mt) * ratio_vec.view(-1, 1)      

        
        # dssp
        y = self.proj1dssp(x_wt2)

        # center-pool
        yresidual = y * cindexvec.view(-1, 1) # sparse vector of 0,1
        yresidual = pyg_nn.global_add_pool(yresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgy = pyg_nn.global_mean_pool(y, batch_mt)

        # relevant-pool v2 (even summation)
        y = y * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        y = pyg_nn.global_add_pool(y, batch_mt) * ratio_vec.view(-1, 1)  
        




        b, c = x.shape
        x = torch.cat([self.cbn1(x), self.cbn3(y),
                       self.cbn4(residual), self.cbn6(yresidual), avgx, avgy], dim=0)

        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)
        x = self.norm5(self.gsa2(x, edge_index_mt))
        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x


################## ablation study 04 #################
# SFM_Netablastu04 without sum&diff, using 12 vectors + attention
######################################################
# SFM_Netablastu04: GSABlockvz3ablastu04
######################################################
class GSABlockvz3ablastu04(torch.nn.Module):
    def __init__(self, dim, node, dropout):
        super(GSABlockvz3ablastu04, self).__init__()
        self.nx1 = nn.LayerNorm(dim, eps=1e-06)
        self.nx2 = nn.LayerNorm(dim, eps=1e-06)
        self.nx3 = nn.LayerNorm(dim, eps=1e-06)
        self.nx4 = nn.LayerNorm(dim, eps=1e-06)
        self.nx5 = nn.LayerNorm(dim, eps=1e-06)
        self.nx6 = nn.LayerNorm(dim, eps=1e-06)
        self.nx7 = nn.LayerNorm(dim, eps=1e-06)
        self.nx8 = nn.LayerNorm(dim, eps=1e-06)
        self.nx9 = nn.LayerNorm(dim, eps=1e-06)
        self.nxq = nn.LayerNorm(dim, eps=1e-06)

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.ffn = mlp(dim, dropout)

        self.proj = nn.Linear(dim * 9, dim)
        #self.proj = nn.Linear(dim, dim)
        self.postq = nn.Linear(dim * 9, dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(0.1)

        self.softmax = nn.Softmax(dim=-1)

        self.g = np.max([dim // 32, 1])
        self.d = dim//self.g
        self.scale = 1./math.sqrt(self.d)

        self.node = node # node = 10

    def forward(self, x, site):
        b, c = x.shape
        g, d = self.g, self.d

        n = b//12

        x = x.view(12, n, c)
        q_ = x[:8, :, :].mean(dim=0)

        x1 = self.nx1(x[0])
        x2 = self.nx2(x[1])
        x3 = self.nx3(x[2])
        x4 = self.nx4(x[3])
        x5 = self.nx5(x[4])
        x6 = self.nx6(x[5])
        x7 = self.nx7(x[6])
        x8 = self.nx8(x[7])
        
        x9 = self.nx9(x[8])
        x10 = self.nx9(x[9])
        x11 = self.nx9(x[10])
        x12 = self.nx9(x[11])
        qn = self.nxq(q_)

        x = torch.cat([qn, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], dim=0).view(13, n, c)

        q = x[:9, :, :] # 6 N c
        q = self.q(q).view(9, n, g, d).permute(1, 2, 0, 3) # n g 3 d
        k = self.k(x).view(13, n, g, d).permute(1, 2, 0, 3) # n g 9 d
        v = self.v(x).view(13, n, g, d).permute(1, 2, 0, 3) # n g 9 d

        sim = self.scale * q @ k.transpose(-2, -1) # 2 6
        sim = self.softmax(sim) # 2 6
        attn = sim @ v # 2

        attn = attn.transpose(1, 2).reshape(n, 9*c)
        #attn = attn.transpose(1, 2).reshape(n, c)

        attn = self.proj(attn)
        attn = self.dropout(attn)

        #q = q.transpose(1, 2).reshape(n, 2*c)
        #q = self.postq(q)
        q = q_
        #x = x.transpose(0, 1).reshape(n, 6*c)
        #q = self.postq(x)
        #q = self.dropout(q)

        x = q + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x
    
    
    
################## ablation study 04 #################
# SFM_Netablastu04 without sum&diff, using 12 vectors + attention
######################################################
# SFM_Netablastu04: GSABlockvz3ablastu04
######################################################
class SFM_Netablastu04 (torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu04 , self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645:SFM_Netablastu04, dropout:{dropout}, dropout_heavy:{dropout_heavy}')        
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        #self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa2 = GSABlock(1024, 10, dropout=dropout)
        #self.gsa3 = GSABlockvz3ablastu04(1024, 10, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout = 0.2)#0801版本将dropout固定为0.1 OR 0.2 or0.3 or0
        self.gsa3 = GSABlockvz3ablastu04(1024, 10, dropout = 0.2)
        
        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        ## PROT5 WT GNN
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)

        ## DSSP GNN
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)

        ## PSSM GNN
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)

        ## PROT5 MT GNN
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)



        # center-pool
        cindexvec = self.prevec.repeat(center_pl_mt.shape)
        cindexvec[center_pl_mt==1.] = 1.
        self.cindexvec = cindexvec





        # relevant-pool v2 (even summation)
        selecttion_vec = self.prevec.repeat(center_pl_mt.shape)
        selecttion_vec[center_pl_mt>0] = 1.
        self.selecttion_vec = selecttion_vec

        ratio_vec = pyg_nn.global_add_pool(selecttion_vec, batch_mt).reciprocal()



        #x = self.gsa2(x, edge_index_mt)


        # WT
        xwt = self.proj1dssp(x_wt1)
        # center-pool
        residualwt = xwt * cindexvec.view(-1, 1) # sparse vector of 0,1
        residualwt = pyg_nn.global_add_pool(residualwt, batch_mt)

        # mean pool
        avgxwt = pyg_nn.global_mean_pool(xwt, batch_mt)
        
        # relevant-pool v2 (even summation)
        xwt = xwt * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        xwt = pyg_nn.global_add_pool(xwt, batch_mt) * ratio_vec.view(-1, 1)

        b, c = xwt.shape
        # MT
        xmt = self.proj1dssp(x_mt)
        # center-pool
        residualmt = xmt * cindexvec.view(-1, 1) # sparse vector of 0,1
        residualmt = pyg_nn.global_add_pool(residualmt, batch_mt)


        # mean pool
        avgxmt = pyg_nn.global_mean_pool(xmt, batch_mt)
        
        # relevant-pool v2 (even summation)
        xmt = xmt * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        xmt = pyg_nn.global_add_pool(xmt, batch_mt) * ratio_vec.view(-1, 1)    
        

        
        # dssp
        y = self.proj1dssp(x_wt2)

        # center-pool
        yresidual = y * cindexvec.view(-1, 1) # sparse vector of 0,1
        yresidual = pyg_nn.global_add_pool(yresidual, batch_mt)


        # mean pool
        avgy = pyg_nn.global_mean_pool(y, batch_mt)


        # relevant-pool v2 (even summation)
        y = y * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        y = pyg_nn.global_add_pool(y, batch_mt) * ratio_vec.view(-1, 1)

        # pssm
        z = self.proj1pssm(x_wt3)

        # center-pool
        zresidual = z * cindexvec.view(-1, 1) # sparse vector of 0,1
        zresidual = pyg_nn.global_add_pool(zresidual, batch_mt)

        #residual = self.gsa1(residual, edge_index_mt)

        # mean pool
        avgz = pyg_nn.global_mean_pool(z, batch_mt)

        # relevant-pool v2 (even summation)
        z = z * selecttion_vec.view(-1, 1) # sparse vec of continuous values
        z = pyg_nn.global_add_pool(z, batch_mt) * ratio_vec.view(-1, 1)

        x = torch.cat([self.cbn1(xwt), self.cbn2(xmt), self.cbn3(y), self.cbn4(z),
                       self.cbn5(residualwt), self.cbn6(residualmt), self.cbn7(yresidual), self.cbn8(zresidual), 
                       avgxwt, avgxmt, avgy, avgz], dim=0)

        x = self.gsa3(x, edge_index_mt)#.view(b, 2, c).mean(dim=1)

        x = self.norm5(self.gsa2(x, edge_index_mt))


        x = self.mlp2(x)
        #x = self.dropout2(x)
        self.v0 = x.shape
        x = x.squeeze()

        return x


################## ablation study 05 ######################
# MLP--attention structure
###########################################################
class SFM_Netablastu05(torch.nn.Module):
    def __init__(self):
        super(SFM_Netablastu05, self).__init__()

        dropout = 0.8

        dropout_heavy = 0.8
        
        print(f'S645:SFM_Netablastu05, dropout:{dropout}, dropout_heavy:{dropout_heavy}')
        self.dropout_in = nn.Dropout(0.)

        #self.prenorm = nn.LayerNorm(1024, eps=1e-06)
        #self.prenorm = nn.BatchNorm1d(1024, eps=1e-06)

        self.conv1 = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2 = GCNConv(nn.Sequential(nn.Linear(1024, 2048),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3 = GCNConv(nn.Sequential(nn.Linear(2048, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residual = GCNConv(nn.Sequential(nn.Linear(1024, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.post = nn.Sequential(silu(), nn.Linear(1024, 1024))


        self.proj1 = nn.Linear(1024, 1024)
        self.proj2 = nn.Linear(1024, 1024)
        self.proj3 = nn.Linear(1024, 1024)



        self.res2 = nn.Linear(1024, 1024)

        self.mlp = nn.Linear(1024+1024, 1024)

        self.gsa1 = GSABlock(1024, 10, dropout=dropout)
        self.gsa2 = GSABlock(1024, 10, dropout=0.2)
        self.gsa3 = GSABlockvz3(1024, 10, dropout=0.2)

        self.mlp2 = nn.Sequential(nn.Linear(1024, 1024),
                                  #nn.SiLU(),
                                  nn.Linear(1024, 1)
                                  )

        self.drop_path = DropPath(0.1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_heavy = nn.Dropout(dropout_heavy)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

        self.norm1 = nn.LayerNorm(1024, eps=1e-06)
        self.norm2 = nn.LayerNorm(1024, eps=1e-06)
        self.norm3 = nn.LayerNorm(1024, eps=1e-06)
        self.norm4 = nn.LayerNorm(1024, eps=1e-06)
        self.norm5 = nn.LayerNorm(1024, eps=1e-06)

        # center-pool index
        self.register_buffer('prevec', torch.zeros(1))

        self.apply(self._init_weights)


        # dssp
        self.conv1dssp = GCNConv(nn.Sequential(nn.Linear(8, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2dssp = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3dssp = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualdssp = GCNConv(nn.Sequential(nn.Linear(8, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        # pssm
        self.conv1pssm = GCNConv(nn.Sequential(nn.Linear(20, 64),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv2pssm = GCNConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.conv3pssm = GCNConv(nn.Sequential(nn.Linear(256, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)
        self.residualpssm = GCNConv(nn.Sequential(nn.Linear(20, 1024),
                                           nn.SiLU()
                                           ), eps=1e-06, train_eps=True)

        self.postdssp = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.postpssm = nn.Sequential(silu(), nn.Linear(1024, 1024))
        self.proj1dssp = nn.Linear(1024, 1024)
        self.proj1pssm = nn.Linear(1024, 1024)


        self.cbn1 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn2 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn3 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn4 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn5 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn6 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)
        self.cbn7 = nn.Identity()# nn.LayerNorm(1024, eps=1e-06)
        self.cbn8 = nn.Identity()#nn.LayerNorm(1024, eps=1e-06)

        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, data_wt, data_mt):
        x_wt1, x_wt2, x_wt3 = data_wt.x[:, 0:1024], data_wt.x[:, 1024:1032], data_wt.x[:, 1032:1052]
        edge_index_wt, batch_wt, center_pl_wt =  data_wt.edge_index, data_wt.batch, data_wt.center_pl
        x_mt, edge_index_mt, batch_mt, center_pl_mt = data_mt.x, data_mt.edge_index, data_mt.batch, data_mt.center_pl
        self.value = batch_wt
        self.value_shape = batch_wt.shape
        self.cenplwt = center_pl_wt
        self.cenplmt = center_pl_mt
        #print(x_wt, x_wt.shape)
        #print(x_mt, x_mt.shape)


        x_wt1 = self.dropout_in(x_wt1)

        x_wt2 = self.dropout_in(x_wt2)

        x_wt3 = self.dropout_in(x_wt3)

        x_mt = self.dropout_in(x_mt)



        ## PROT5 WT GNN
        residual = x_wt1
        res_index = edge_index_wt
        x_wt1 = self.conv1(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv2(x_wt1, edge_index_wt)
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout(x_wt1)

        x_wt1 = self.conv3(x_wt1, edge_index_wt)
        x_wt1 = x_wt1 + self.post(self.residual(residual, res_index))
        x_wt1 = F.silu(x_wt1)

        x_wt1 = self.dropout_heavy(x_wt1)

        ## DSSP GNN
        residual = x_wt2
        res_index = edge_index_wt
        x_wt2 = self.conv1dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv2dssp(x_wt2, edge_index_wt)
        x_wt2 = F.silu(x_wt2)

        #x_wt2 = self.dropout(x_wt2)

        x_wt2 = self.conv3dssp(x_wt2, edge_index_wt)
        x_wt2 = x_wt2 + self.postdssp(self.residualdssp(residual, res_index))
        x_wt2 = F.silu(x_wt2)

        x_wt2 = self.dropout_heavy(x_wt2)

        ## PSSM GNN
        residual = x_wt3
        res_index = edge_index_wt
        x_wt3 = self.conv1pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv2pssm(x_wt3, edge_index_wt)
        x_wt3 = F.silu(x_wt3)

        #x_wt3 = self.dropout(x_wt3)

        x_wt3 = self.conv3pssm(x_wt3, edge_index_wt)
        x_wt3 = x_wt3 + self.postpssm(self.residualpssm(residual, res_index))
        x_wt3 = F.silu(x_wt3)

        x_wt3 = self.dropout_heavy(x_wt3)

        ## PROT5 MT GNN
        residual = x_mt
        res_index = edge_index_mt
        x_mt = self.conv1(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv2(x_mt, edge_index_mt)
        x_mt = F.silu(x_mt)

        x_mt = self.dropout(x_mt)

        x_mt = self.conv3(x_mt, edge_index_mt)
        x_mt = x_mt + self.post(self.residual(residual, res_index))
        x_mt = F.silu(x_mt)

        x_mt = self.dropout_heavy(x_mt)

        # SUM & DIFF
        x = torch.cat([self.proj1(x_wt1), self.proj1(x_mt)], dim=1)

        residual = self.proj2(x_mt - x_wt1)

        residual = self.res2(residual)

        x = self.mlp(x)

        x = self.drop_path(self.norm2(residual)) + self.norm1(x)

        x = F.silu(x)

        x = self.dropout_heavy(x)

        x = self.proj3(x)



        # mean pool
        avgx = pyg_nn.global_mean_pool(x, batch_wt)
        # mean pool
        avgx_wt2 = pyg_nn.global_mean_pool(x_wt2, batch_wt)
        # mean pool
        avgx_wt3 = pyg_nn.global_mean_pool(x_wt3, batch_wt)        
        #print(avgx_wt1.shape, avgx_mt.shape, avgx_wt2.shape, avgx_wt3.shape)
        
        x = torch.cat([self.cbn1(avgx), self.cbn2(avgx_wt2), self.cbn3(avgx_wt3)], dim=0)
        #print(x.shape)
        b, c = avgx.shape
        x = x.view(3, b, c)
        #print(x.shape)
        x = torch.mean(x, dim=0, keepdim=True)
        #print(x.shape)
        x = self.mlp2(x)
        #print(x.shape)

        self.v0 = x.shape
        x = x.squeeze()

        return x
