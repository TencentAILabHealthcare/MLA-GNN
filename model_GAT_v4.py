
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling
from torch.nn import LayerNorm, Parameter
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.utils import to_dense_batch, to_dense_adj
from utils import *


class GAT(torch.nn.Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.fc_dropout = opt.fc_dropout
        self.GAT_dropout = opt.GAT_dropout
        self.act = define_act_layer(act_type=opt.act_type)

        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [64, 48, 32]

        self.conv1 = GATConv(opt.input_dim, self.nhids[0], heads=self.nheads[0],
                             dropout=self.GAT_dropout)
        self.conv2 = GATConv(self.nhids[0]*self.nheads[0], self.nhids[1], heads=self.nheads[1],
                             dropout=self.GAT_dropout)
        self.conv3 = GATConv(self.nhids[1]*self.nheads[1], self.nhids[2], heads=self.nheads[2],
                             dropout=self.GAT_dropout)

        self.pool1 = torch.nn.Linear(self.nhids[0]*self.nheads[0], 1)
        self.pool2 = torch.nn.Linear(self.nhids[1]*self.nheads[1], 1)
        self.pool3 = torch.nn.Linear(self.nhids[2]*self.nheads[2], 1)

        self.layer_norm0 = LayerNorm(opt.num_nodes)
        self.layer_norm1 = LayerNorm(opt.num_nodes)
        self.layer_norm2 = LayerNorm(opt.num_nodes)
        self.layer_norm3 = LayerNorm(opt.num_nodes)

        fc1 = nn.Sequential(
            nn.Linear(opt.lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=True))

        fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False))

        fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False))

        fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], opt.omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.fc_dropout, inplace=False))

        self.encoder = nn.Sequential(fc1, fc2, fc3, fc4)
        self.classifier = nn.Sequential(nn.Linear(opt.omic_dim, opt.label_dim))

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


    def forward(self, x, adj, grad_labels, batch, opt):

        ### layer1
        x = x.requires_grad_()
        x0 = to_dense_batch(torch.mean(x, dim=-1), batch=batch)[0] #[bs, nodes]

        ### layer2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, adj)) #[bs*nodes, nhids[0]*nheads[0]]

        x1 = to_dense_batch(self.pool1(x).squeeze(-1), batch=batch)[0] #[bs, nodes]

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, adj))  # [bs*nodes, nhids[0]*nheads[0]]

        x2 = to_dense_batch(self.pool2(x).squeeze(-1), batch=batch)[0]  # [bs, nodes]


        if opt.layer_norm == "True":
            x0 = self.layer_norm0(x0)
            x1 = self.layer_norm1(x1)
            x2 = self.layer_norm0(x2)

        if opt.which_layer == 'all':
            x = torch.cat([x0, x1, x2], dim=1)

        elif opt.which_layer == 'layer1':
            x = x0
        elif opt.which_layer == 'layer2':
            x = x1
        elif opt.which_layer == 'layer3':
             x = x2

        GAT_features = x

        features = self.encoder(x)
        out = self.classifier(features)

        fc_features = features

        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift


        return GAT_features, fc_features, out


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