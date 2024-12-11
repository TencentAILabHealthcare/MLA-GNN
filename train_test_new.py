import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import logging
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import torch.nn.functional as F

from utils import *
from model_GAT_v4 import *


def train(opt, train_dataset, test_dataset):

    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2020)
    torch.manual_seed(2020)
    random.seed(2020)

    model = MLA_GNN(opt).cuda()
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=8)
    metric_logger = {'train': {'loss': [], 'grad_acc': []}, 'test': {'loss': [], 'grad_acc': []}}

    # print("============ finish dataloader ===========")


    for epoch in range(opt.num_epochs):
        model.train()
        loss_epoch, grad_acc_epoch = 0, 0

        class_edge_weights = torch.zeros(opt.label_dim, opt.num_nodes, opt.num_nodes).cuda()
        class_node_importance = torch.zeros(opt.label_dim, opt.num_nodes).cuda()
        overall_edge_weights = torch.zeros(opt.num_nodes, opt.num_nodes).cuda()
        overall_node_importance = torch.zeros(opt.num_nodes).cuda()
        # print(class_edge_weights.shape, class_node_importance.shape)
        # print(overall_edge_weights.shape, overall_node_importance.shape)

        for batch_idx, data in enumerate(train_loader):
            batch_idx += 1
            tr_features, tr_fc_features, tr_preds, edge_weights, feature_importance = model(
                data.x.cuda(), data.edge_index.cuda(), data.y.cuda(), data.batch.cuda(), opt)
            # print(data.y)
            sample_weight = torch.tensor(cal_sample_weight(
                data.y, opt.label_dim, use_sample_weight=True)).cuda()
            # print("===========", data.y, sample_weight)

            """
            Compute the edge weights and node importance for each class.
            """
            overall_edge_weights += torch.mean(edge_weights, 0)
            overall_node_importance += torch.mean(feature_importance, 0)

            for i in range(opt.label_dim):
                index = np.nonzero(data.y == i).view(1, -1)[0]
                # print(torch.index_select(edge_weights, 0, index[0].cuda()).shape)
                if index.shape[0] > 0:
                    # print(index.shape[0])
                    class_edge_weights[i] += torch.mean(
                        torch.index_select(edge_weights, 0, index.cuda()), 0)
                    class_node_importance[i] += torch.mean(
                        torch.index_select(feature_importance, 0, index.cuda()), 0)
                    # print(class_node_importance[i])

            loss_reg = define_reg(model)
            loss_func = nn.CrossEntropyLoss(reduction='none')
            grad_loss = torch.mean(torch.mul(loss_func(tr_preds, data.y.cuda()), sample_weight))
            # print(grad_loss)
            # one_hot_labels = one_hot_tensor(batch_labels.cuda(), opt.label_dim)
            # grad_loss = torch.mean(torch.mul(tr_preds - one_hot_labels, tr_preds - one_hot_labels))
            loss = opt.lambda_nll * grad_loss + opt.lambda_reg * loss_reg
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            # tr_features.retain_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            pred = tr_preds.argmax(dim=1, keepdim=True)
            grad_acc_epoch += pred.eq(data.y.cuda().view_as(pred)).sum().item()

        scheduler.step()

        class_edge_weights = (class_edge_weights/batch_idx).detach().cpu().numpy()
        class_node_importance = (class_node_importance/batch_idx).detach().cpu().numpy()
        overall_edge_weights = (overall_edge_weights/batch_idx).detach().cpu().numpy()
        overall_node_importance = (overall_node_importance/batch_idx).detach().cpu().numpy()

        # print(gradients_all.shape, importance_all.shape)
        # print(class_edge_weights, class_node_importance)

        loss_epoch /= len(train_loader.dataset)
        grad_acc_epoch = grad_acc_epoch / len(train_loader.dataset)
        loss_test, grad_acc_test, pred_test, features_test, _ = test(opt, model, test_dataset)
        # loss_test, grad_acc_test, pred_test, features_test, _ = test(opt, model, train_dataset)

        metric_logger['train']['loss'].append(loss_epoch)
        metric_logger['train']['grad_acc'].append(grad_acc_epoch)

        metric_logger['test']['loss'].append(loss_test)
        metric_logger['test']['grad_acc'].append(grad_acc_test)

        # pickle.dump(pred_test, open(os.path.join(
        #     opt.results_dir, 'split%d_%d_pred_test.pkl' % (k, epoch)), 'wb'))


        logging.info('\nEpoch {:02d}/{:02d}, [{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format(
            epoch + 1, opt.num_epochs, 'Train', loss_epoch, 'Accuracy', grad_acc_epoch))
        logging.info('\nEpoch {:02d}/{:02d}, [{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format(
            epoch + 1, opt.num_epochs, 'Test', loss_test, 'Accuracy', grad_acc_test))

        # print("=========gradients_all:", gradients_all)

    return model, optimizer, metric_logger, class_edge_weights, class_node_importance, \
           overall_edge_weights, overall_node_importance


def test(opt, model, test_dataset):

    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (data) in enumerate(test_loader):

        te_features, te_fc_features, te_preds, _, _ = model(
            data.x.cuda(), data.edge_index.cuda(), data.y.cuda(), data.batch.cuda(), opt)

        # print(te_preds, te_fc_features)
        # print(data.y, torch.argmax(te_preds, dim=1))
        # print(data.y, te_preds)
        # print("te_preds:", te_preds)

        if batch_idx == 0:
            features_all = te_features.detach().cpu().numpy()
            fc_features_all = te_fc_features.detach().cpu().numpy()
        else:
            features_all = np.concatenate((features_all, te_features.detach().cpu().numpy()), axis=0)
            fc_features_all = np.concatenate((fc_features_all, te_fc_features.detach().cpu().numpy()), axis=0)
        # print(features_all.shape, te_features.shape)

        loss_reg = define_reg(model)
        loss_func = nn.CrossEntropyLoss()
        grad_loss = loss_func(te_preds, data.y.cuda())
        # one_hot_labels = one_hot_tensor(batch_labels.cuda(), opt.label_dim)
        # grad_loss = torch.mean(torch.mul(te_preds - one_hot_labels, te_preds - one_hot_labels))
        loss = opt.lambda_nll * grad_loss + opt.lambda_reg * loss_reg
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, data.y.reshape(-1)))  # Logging Information

        pred = te_preds.argmax(dim=1, keepdim=True)
        grad_acc_test += pred.eq(data.y.cuda().view_as(pred)).sum().item()
        probs_np = te_preds.detach().cpu().numpy()
        probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)  # Logging Information

    # print("total batch:", batch_idx)

    # print(survtime_all)
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    grad_acc_test = grad_acc_test / len(test_loader.dataset)
    pred_test = [probs_all, gt_all]
    # print(probs_all.shape, gt_all.shape)

    return loss_test, grad_acc_test, pred_test, features_all, fc_features_all




def external_test(opt, model, test_dataset):

    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
    probs_all = None

    for batch_idx, (data) in enumerate(test_loader):

        te_features, te_fc_features, te_preds, _, _ = model(
            data.x.cuda(), data.edge_index.cuda(), data.y.cuda(), data.batch.cuda(), opt)
        # print("features:", te_features)
        probs_np = F.softmax(te_preds, 1).detach().cpu().numpy()
        probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)

    # print(probs_all)

    return probs_all

