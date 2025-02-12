from torch_geometric.nn import GINConv, HypergraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph
import copy
from math import sqrt


def Subgraph(data, aug_ratio):
    data = copy.deepcopy(data)

    # return data
    x = data.x
    edge_index = data.edge_index

    sub_num = int(data.num_nodes * aug_ratio)
    idx_sub = torch.randint(0, data.num_nodes, (1,)).to(edge_index.device)
    last_idx = idx_sub
    diff = None

    for k in range(1, sub_num):
        keep_idx, _, _, _ = k_hop_subgraph(last_idx, 1, edge_index)
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = torch.cat((last_idx, keep_idx)).to(edge_index.device)
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = torch.randperm(diff.shape[0])[:diff_keep_num].to(edge_index.device)
    final_keep_idx = torch.cat((last_idx, diff_keep_idx))

    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0

    final_keep_idx = final_keep_idx

    edge_index, _ = subgraph(final_keep_idx, edge_index)

    data.x = x
    data.edge_index = edge_index
    return data



class HyperGF(nn.Module):
    def __init__(self, num_classes=1,num_features_xd=84,dropout=0.2,aug_ratio=0.4,hidden_feats=None,num_heads=4,fp_2_dim=128,fp_dim = 2513):
        super(HyperGF, self).__init__()
        if hidden_feats is None:
            hidden_feats = [512, 512]
        self.final_hidden_feats = hidden_feats[-1]
        self.fp_dim = fp_dim
        self.fp_2_dim = fp_2_dim
        self.fc1 = nn.Linear(2513, self.fp_2_dim)
        self.dropout_fpn = dropout
        self.act_func = nn.ReLU()
        self.fp_2_dim = fp_2_dim
        self.fc2 = nn.Linear(self.fp_2_dim, self.final_hidden_feats)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)

        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.Linear(512, 512)
        )

        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd*10*2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_h_g = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        #self.fplinear = nn.Linear(2513, 512)
        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.relu = nn.ReLU()
        self.aug_ratio = aug_ratio

        self.conv3 = HypergraphConv(num_features_xd, num_features_xd*2)
        self.conv4 = HypergraphConv(num_features_xd*2, 256)

        self.fc_final = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # if hidden_feats is None:
        #     hidden_feats = [1536, 1536]



        dim_k = self.final_hidden_feats * num_heads
        dim_v = self.final_hidden_feats * num_heads
        dim_in = self.final_hidden_feats
        assert dim_k % 4 == 0 and dim_v % 4 == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.mlp = nn.Sequential(
            nn.Linear((self.final_hidden_feats - 2) * self.num_heads, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3), nn.ReLU(),
                                  nn.Dropout(dropout))


    def forward(self, data,x,edge_index,batch,hedge_index,batch_size,fp):
        # fingerprint
        fp = fp.view(batch_size, -1)
        fp_x = self.fc1(fp)
        fp = self.dropout(fp_x)
        fp_x = self.act_func(fp)
        fp = self.fc2(fp_x)


        # Graph Encoder
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))
        x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        x_g = self.fc_g(x_g)


        #hypergarph
        hnode = self.conv3(x, hedge_index)
        hnode = self.conv4(hnode, hedge_index)
        hx_g = torch.cat([gmp(hnode, batch), gap(hnode, batch)], dim=1)
        hx_g = self.fc_h_g(hx_g)

        # Stability predictor

        fp=fp.unsqueeze(1)
        x_g1=x_g.unsqueeze(1)
        hx_g1=hx_g.unsqueeze(1)
        in_tensor = torch.cat([fp, x_g1, hx_g1], dim=1)

        batch, n, dim_in = in_tensor.shape
        #assert dim_in == self.dim_in
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        q = self.linear_q(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(in_tensor).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        att = torch.matmul(dist, v)



        out = self.conv(att).view(batch_size, -1)
        out = self.mlp(out)
        return out,x_g,hx_g

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)







