import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold

from utils import *
from model_GAT import *



def test(opt, model, te_features, te_labels, adj_matrix):

    model.eval()

    test_dataset = Data.TensorDataset(te_features, te_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (batch_features, batch_labels) in enumerate(test_loader):
        censor = batch_labels[:, 0]
        survtime = batch_labels[:, 1]
        grade = batch_labels[:, 2]
        censor_batch_labels = censor.cuda() if "surv" in opt.task else censor
        surv_batch_labels = survtime
        # print(surv_batch_labels)
        grad_batch_labels = grade.cuda() if "grad" in opt.task else grade
        te_features, te_fc_features, te_preds, gradients, feature_importance = model(
            batch_features.cuda(), adj_matrix.cuda(), grad_batch_labels, opt)

        # print("surv_batch_labels:", surv_batch_labels)
        # print("te_preds:", te_preds)

        if batch_idx == 0:
            features_all = te_features.detach().cpu().numpy()
            fc_features_all = te_fc_features.detach().cpu().numpy()
        else:
            features_all = np.concatenate((features_all, te_features.detach().cpu().numpy()), axis=0)
            fc_features_all = np.concatenate((fc_features_all, te_fc_features.detach().cpu().numpy()), axis=0)
        # print(features_all.shape, te_features.shape)

        loss_cox = CoxLoss(surv_batch_labels, censor_batch_labels, te_preds) if opt.task == "surv" else 0
        loss_reg = define_reg(model)
        loss_func = nn.CrossEntropyLoss()
        grad_loss = loss_func(te_preds, grad_batch_labels) if opt.task == "grad" else 0
        loss = opt.lambda_cox * loss_cox + opt.lambda_nll * grad_loss + opt.lambda_reg * loss_reg
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, grad_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information

        if opt.task == "surv":
            risk_pred_all = np.concatenate((risk_pred_all, te_preds.detach().cpu().numpy().reshape(-1)))  # Logging Information
            censor_all = np.concatenate((censor_all, censor_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information
            survtime_all = np.concatenate((survtime_all, surv_batch_labels.detach().cpu().numpy().reshape(-1)))  # Logging Information

        elif opt.task == "grad":
            pred = te_preds.argmax(dim=1, keepdim=True)
            grad_acc_test += pred.eq(grad_batch_labels.view_as(pred)).sum().item()
            probs_np = te_preds.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)  # Logging Information

    # print(survtime_all)
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, features_all, fc_features_all