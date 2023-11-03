import torch
import torch.nn as nn
import math
import copy
import random
import numpy as np
import torch.nn.functional as F

from modules import ConvLayer, ReconstructionModel
from transformer import EncoderLayer, MultiHeadAttention
from graph import GraphEmbedding


'''

results, 2graph,  win=8  :
(F1, P, R)
--msl: (95.08,93.48,96.73)--20230324_104633   2.7   30 epoch
--smap:(90.32,95.62,85.58)--20230324_105432   6.0   30
--swat:(90.19,93.72,86.92)--20230331_172744   4.5   30
--wadi:(89.76,91.78,87.84)--20230329_105629   6.9   30
-------------------------------------------
results, 2graph,  win=16  :              
(F1, P, R)
--msl: (95.49,95.10,95.89)--20230323_173247   3.1
--smap:(91.83,90.81,92.88)--20230311_220757   6.0
--swat:(91.19,96.76,86.23)--20230322_102823   4.8
--wadi:(92.64,98.00,87.84)--20230329_095534   9.3
-------------------------------------------
results, 2graph,  win=32  :
(F1, P, R)
--msl: (95.00,94.18,95.84)--20230321_102427   4.8
--smap:(93.07,93.42,92.71)--20230327_171113   7.8
--swat:(91.87,94.81,89.11)--20230321_134845   7.5
--wadi:(88.31,88.79,87.84)--20230329_111323   18.0
--------------------------------------------
results, 2graph,  win=64  :
(F1, P, R)
--msl: (93.47,93.55,93.39)--20230320_094825    time:9.7
--smap:(90.30,95.55,85.60)--20230320_171625    time:11.0
--swat:(91.11,93.20,89.11)--20230331_174459    time:14.4
--wadi:(86.90,85.99,87.84)--20230317_180052    time:39.6
--------------------------------------------

---sensitivity---
graph layer=1  msl   (95.49,96.10,95.89)--20230323_173247     3.1
graph layer=2  msl   (94.34,94.82,93.86)--20230330_114206     4.3
graph layer=3  msl   (93.98,96.80,91.15)--20230329_165015     5.4
graph layer=4  msl   (93.49,93.90,93.10)--20230329_165631     6.6
graph layer=5  msl   (94.97,93.53,96.46)--20230329_171313     7.5
graph layer=6  msl   (93.08,93.09,93.09)--20230329_173156     8.8
graph layer=7  msl   (92.36,94.44,90.37)--20230329_174651     9.8
graph layer=8  msl   (92.81,92.14,93.48)--20230329_180412     10.8


---ablation---
results, 2graph cl, without transformer (use rnn)  win=16 
(F1, P, R)
--msl: (94.18,94.23,94.13)--20230328_150712    time:3.0    
--smap:(91.37,97.76,85.60)--20230328_171742    time:5.8
--swat:(89.62,90.14,89.11)--20230328_170613    time:4.7
--wadi:(86.29,84.80,87.84)--20230329_104913    9.5
--------------------------------------------

results, 2graph cl, without cross loss  win=16  :
(F1, P, R)
--msl: (94.55,92.35,96.86)--20230328_171229       2.8
--smap:(88.69,95.41,82.86)--20230329_103604       5.2
--swat:(89.15,94.98,83.99)--20230329_101612       4.4
--wadi:(90.76,93.88,87.84)--20230328_164119       9.0



results, 2 graph, without Contrastive learning  win=16  :
(F1, P, R)
--msl: (91.19,93.97,88.55)--20230329_123018  time: 2.1
--smap:(88.24,94.46,82.79)--20230329_123931      4.1
--swat:(89.27,95.25,83.99)--20230329_123437      3.3
--wadi:(88.79,89.76,87.84)--20230329_112802      7.2


results, only using intra graph cl win=16  :
(F1, P, R)
--msl: (90.96,91.96,89.98)--20230329_141006  time: 1.9
--smap:(87.71,96.50,80.38)--20230329_141727        4.2
--swat:(87.84,95.78,81.12)--20230329_143828        3.0
--wadi:(85.84,83.93,87.84)--20230329_144515   4.4


results, only using inter graph cl win=16  :
(F1, P, R)
--msl: (84.57,76.78,94.09)--20230329_152426   2.2
--smap:(80.80,86.91,75.49)--20230329_151117   4.0
--swat:(86.83,99.04,77.30)--20230329_150300   3.4
--wadi:(80.10,73.61,87.84)--20230329_145626   7.7



'''
# import torch.nn.functional as F

# def sim(h1, h2):
#     z1 = F.normalize(h1, dim=-1, p=2)
#     z2 = F.normalize(h2, dim=-1, p=2)
#     return torch.mm(z1, z2.t())

# def contrastive_loss(h1, h2):
#     f = lambda x: torch.exp(x)
#     inter_sim = f(sim(h1, h2))
#     return -torch.log(inter_sim.diag() /
#                      (inter_sim.sum(dim=-1) - inter_sim.diag()))


class ProjectionLayer(nn.Module):
    """Encoder的一层"""

    def __init__(self, n_feature, num_heads, dropout=0.1):
        super(ProjectionLayer, self).__init__()

        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)

    def forward(self, inputs, attn_mask=None):

        context, _ = self.attention(inputs, inputs, inputs, attn_mask)
        return context


class MODEL_CGRAPH_TRANS(nn.Module):
    """ MODEL_CGRAPH_TRANS model class.
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        dropout=0.2,
        device = 'cuda:0'
    ):
        super(MODEL_CGRAPH_TRANS, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim
        self.f_dim = n_features
        self.decoder_type = 0  # 0:transformer   1:rnn
        self.use_cross_loss = True  # 
        self.use_contrastive = True
        self.use_intra_graph = True  
        self.use_inter_graph = False

        # preprocessing
        self.conv = ConvLayer(n_features, 7)

        # augmentation learnable parameters
        self.param = nn.Linear(self.f_dim, self.f_dim)

        self.num_levels = 1
        # inter embedding module based on GNN
        self.inter_module = GraphEmbedding(num_nodes=n_features, seq_len=window_size, num_levels=self.num_levels, device=torch.device(device))

        # intra embedding module based on GNN
        self.intra_module = GraphEmbedding(num_nodes=window_size, seq_len=n_features, num_levels=self.num_levels, device=torch.device(device))

        # projection head
        #self.proj_head_inter = ProjectionLayer(n_feature=self.f_dim, num_heads=1, dropout=dropout)
        self.proj_head_inter = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.f_dim)
        )

        # projection head
        #self.proj_head_intra = ProjectionLayer(n_feature=window_size, num_heads=1, dropout=dropout)
        self.proj_head_intra = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, window_size)
        )

        # projection head
        self.proj_head3 = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.f_dim)
        )

        self.fusion_linear = nn.Linear(self.f_dim*2, self.f_dim)

        self.inter_latent = None 
        self.intra_latent = None
        self.fused_latent = None

        # decoder
        if self.decoder_type == 0:
            self.decoder = EncoderLayer(n_feature=self.f_dim, num_heads=1, hid_dim=self.f_dim, dropout=dropout)
        elif self.decoder_type == 1:
            self.decoder = ReconstructionModel(in_dim=self.f_dim, hid_dim=100, out_dim=self.f_dim, n_layers=1, dropout=dropout)
        self.linear = nn.Linear(self.f_dim, self.out_dim)


    def aug_feature1(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
        #drop_dim = np.random.randint(1, 3)
        #print('drop_dim:', drop_dim)
        if drop_dim == 1:
            aug_input_feat[:, drop_idx, :] = 0.
        elif drop_dim == 2:
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature2(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            drop_idx = random.sample([i for i in range(total_num)], drop_feat_num)
            aug_input_feat[:, :, drop_idx] = 0.

        return aug_input_feat


    def aug_feature3(self, input_feat, drop_dim = 2, drop_percent = 0.1):
        aug_input_feat = copy.deepcopy(input_feat)
        drop_dim = random.randint(1, 2)
        total_num = aug_input_feat.shape[drop_dim]
        drop_feat_num = int(total_num * drop_percent)
        # mask 
        if drop_dim == 1:
            ind = total_num - drop_feat_num
            aug_input_feat[:, ind:, :] = 0.
        elif drop_dim == 2:
            p = self.param(aug_input_feat)
            sp = F.gumbel_softmax(p, hard=True)
            aug_input_feat = sp*input_feat

        return aug_input_feat


    def loss_cl_s(self, z1, z2):
        batch_size, w, k = z1.size()
        T = 0.5
        x1 = z1.contiguous().view(batch_size, -1)
        x2 = z2.contiguous().view(batch_size, -1).detach()
        
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


    def loss_cl(self, z1, z2):
        batch_size, w, k = z1.size()
        T = 0.5
        x1 = z1.contiguous().view(batch_size, -1)
        x2 = z2.contiguous().view(batch_size, -1)
        
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_a = torch.exp(sim_matrix_a / T)
        pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
        loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
        loss_a = - torch.log(loss_a).mean()

        sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
        sim_matrix_b = torch.exp(sim_matrix_b / T)
        pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
        loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
        loss_b = - torch.log(loss_b).mean()

        loss = (loss_a + loss_b) / 2
        return loss


    def forward(self, x, training=True):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        #x = self.conv(x)

        if training:
            x_aug = self.aug_feature(x)
        else:
            x_aug = x
        
        # intra graph
        if self.use_intra_graph:
            enc_intra = self.intra_module(x.permute(0, 2, 1))   # >> (b, k, n)
            # projection head
            enc_intra = self.proj_head_intra(enc_intra).permute(0, 2, 1)

        # inter graph
        if self.use_inter_graph:
            enc_inter = self.inter_module(x)  # >> (b, n, k)
            # projection head
            enc_inter = self.proj_head_inter(enc_inter)

        if training and self.use_contrastive:
            if self.use_intra_graph:
                # intra aug
                enc_intra_aug = self.intra_module(x_aug.permute(0, 2, 1))  # >> (b, k, n)
                # projection head
                enc_intra_aug = self.proj_head_intra(enc_intra_aug).permute(0, 2, 1)
                # contrastive loss
                loss_intra_in = self.loss_cl_s(enc_intra, enc_intra_aug)

            if self.use_inter_graph:
                # inter aug
                enc_inter_aug = self.inter_module(x_aug)
                # projection head
                enc_inter_aug = self.proj_head_inter(enc_inter_aug)
                # contrastive loss
                loss_inter_in = self.loss_cl_s(enc_inter, enc_inter_aug)

        # projection head 3
        # enc_intra = self.proj_head3(enc_intra)
        # enc_inter = self.proj_head3(enc_inter)

        # contrastive loss
        if training and self.use_contrastive:
            loss_cl = 0
            if self.use_intra_graph:
                loss_cl += loss_intra_in
            if self.use_inter_graph:
                loss_cl += loss_inter_in
            if self.use_cross_loss and self.use_intra_graph and self.use_inter_graph:
                loss_cross = self.loss_cl(enc_intra, enc_inter)
                loss_cl += loss_cross
        else:
            loss_cl = torch.zeros([1]).cuda()

        # fuse
        if self.use_intra_graph and self.use_inter_graph:
            enc = torch.cat([enc_inter, enc_intra], dim=-1)
            enc = self.fusion_linear(enc)
            # =============
            self.inter_latent = enc_inter
            self.intra_latent = enc_intra
            self.fused_latent = enc 
            # =============
        elif self.use_intra_graph:
            enc = enc_intra
        elif self.use_inter_graph:
            enc = enc_inter

        # decoder
        if self.decoder_type == 0:
            dec, _ = self.decoder(enc)
        elif self.decoder_type == 1:
            dec = self.decoder(enc)
        out = self.linear(dec)

        return out, loss_cl


