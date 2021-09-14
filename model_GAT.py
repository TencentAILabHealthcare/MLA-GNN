"""
This is the implementation of Graph Attention Network.
The code is inspired by "https://github.com/Diego999/pyGAT"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap

from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler

from utils import *


class GAT(nn.Module):
    def __init__(self, opt, input_dim, omic_dim, label_dim, dropout, alpha):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.act = define_act_layer(act_type=opt.act_type)

        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [64, 48, 32]

        self.attentions1 = [GraphAttentionLayer(
            input_dim, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)

        self.attentions2 = [GraphAttentionLayer(
            self.nhids[0]*self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)

        self.attentions3 = [GraphAttentionLayer(
            self.nhids[1]*self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[2])]
        for i, attention3 in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention3)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # lin_input_dim = self.nhids[0]*self.nheads[0] + self.nhids[1]*self.nheads[1] + self.nhids[2]*self.nheads[2]
        lin_input_dim = opt.lin_input_dim

        # self.lin1 = torch.nn.Linear(lin_input_dim, lin_dim1)
        # self.lin2 = torch.nn.Linear(lin_dim1, label_dim)

        self.pool1 = torch.nn.Linear(self.nhids[0]*self.nheads[0], 1)
        self.pool2 = torch.nn.Linear(self.nhids[1]*self.nheads[1], 1)
        self.pool3 = torch.nn.Linear(self.nhids[2] * self.nheads[2], 1)

        fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))

        fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))

        fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))

        fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))

        self.encoder = nn.Sequential(fc1, fc2, fc3, fc4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))


        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


    def forward(self, x, adj, grad_labels, opt):

        # print("input shape:", x.shape)
        batch = torch.linspace(0, x.size(0) - 1, x.size(0), dtype=torch.long)
        batch = batch.unsqueeze(1).repeat(1, x.size(1)).view(-1).cuda()

        if opt.cnv_dim == 80:
            cnv_feature = torch.mean(x[:, :80, :], dim=-1)
        x = x[:, 80:, :]
        x0 = torch.mean(x, dim=-1)
        # print("x0:", x0.shape)

        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1) # [bs, N, nhid1*nhead1]

        x1 = self.pool1(x).squeeze(-1)
        # print("x1:", x1.shape)

        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)  # [bs, N, nhid2*nhead2]

        x2 = self.pool2(x).squeeze(-1)
        # print("x2:", x2)


        if opt.lin_input_dim == 800 or opt.lin_input_dim == 720:
            x = torch.cat([x0, x1, x2], dim=1)
        elif opt.lin_input_dim == 320 or opt.lin_input_dim == 240:
            if opt.which_layer == 'layer1':
                x = x0
            elif opt.which_layer == 'layer2':
                x = x1
            elif opt.which_layer == 'layer3':
                x = x2

        if opt.cnv_dim == 80:
            x = torch.cat([cnv_feature, x], dim=1)

        GAT_features = x

        # print("feature shape:", x.shape)

        features = self.encoder(x)
        out = self.classifier(features)

        fc_features = features

        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        if opt.task == "grad":
            one_hot_labels = torch.zeros(grad_labels.shape[0], 3).cuda().scatter(1, grad_labels.reshape(-1, 1), 1)
            y_c = torch.sum(one_hot_labels*out)
        elif opt.task == "surv":
            y_c = torch.sum(out)
        # print(out, y_c)
        GAT_features.grad = None
        GAT_features.retain_grad()
        y_c.backward(retain_graph=True)
        gradients = np.maximum(GAT_features.grad.detach().cpu().numpy(), 0)# (batch_size, 720)
        feature_importance = np.mean(gradients, 0)

        return GAT_features, fc_features, out, gradients, feature_importance



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input, adj):
        """
        input: mini-batch input. size: [batch_size, num_nodes, node_feature_dim]
        adj:   adjacency matrix. size: [num_nodes, num_nodes].  need to be expanded to batch_adj later.
        """
        h = torch.matmul(input, self.W)# [bs, N, F]
        bs, N, _ = h.size()

        a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1, 2 * self.out_features)
        # print("h size:", a_input.shape)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        batch_adj = torch.unsqueeze(adj, 0).repeat(bs, 1, 1)
        # print("batch adj size:", batch_adj.shape)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(batch_adj > 0, e, zero_vec)
        attention = self.dropout_layer(F.softmax(attention, dim=-1)) # [bs, N, N]
        # print("attention shape:", attention.shape)
        h_prime = torch.bmm(attention, h)# [bs, N, F]
        # print("h_prime:", h_prime.shape)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_reg(model):

    for W in model.parameters():
        loss_reg = torch.abs(W).sum()

    return loss_reg


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1) / float(opt.num_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

