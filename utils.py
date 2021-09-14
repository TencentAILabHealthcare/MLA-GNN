import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset

# import lifelines
# from lifelines.utils import concordance_index
# from lifelines.statistics import logrank_test

from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer

################
# Data Utils
################

def load_csv_data(k, opt):
    folder_path = './example_data/input_features_labels/split'
    print("Loading data from:", folder_path+str(k))
    train_data_path = folder_path+str(k)+'_train_320d_features_labels.csv'
    train_data = np.array(pd.read_csv(train_data_path, header=None))[1:, 2:].astype(float)

    tr_features = torch.FloatTensor(train_data[:, :320].reshape(-1, 320, 1)).requires_grad_()
    tr_labels = torch.LongTensor(train_data[:, 320:])
    print("Training features and labels:", tr_features.shape, tr_labels.shape)

    test_data_path = folder_path+str(k)+'_test_320d_features_labels.csv'
    test_data = np.array(pd.read_csv(test_data_path, header=None))[1:, 2:].astype(float)

    te_features = torch.FloatTensor(test_data[:, :320].reshape(-1, 320, 1)).requires_grad_()
    te_labels = torch.LongTensor(test_data[:, 320:])
    print("Testing features and labels:", te_features.shape, te_labels.shape)

    similarity_matrix = np.array(pd.read_csv(
        './example_data/input_adjacency_matrix/split'+str(k)+'_adjacency_matrix.csv')).astype(float)
    adj_matrix = torch.LongTensor(np.where(similarity_matrix > opt.adj_thresh, 1, 0))
    print("Adjacency matrix:", adj_matrix.shape)
    print("Number of edges:", adj_matrix.sum())

    if opt.task == "grad":
        tr_idx = tr_labels[:, 2] >= 0
        tr_labels = tr_labels[tr_idx]
        tr_features = tr_features[tr_idx]
        print("Training features and grade labels after deleting NA labels:", tr_features.shape, tr_labels.shape)

        te_idx = te_labels[:, 2] >= 0
        te_labels = te_labels[te_idx]
        te_features = te_features[te_idx]
        print("Testing features and grade labels after deleting NA labels:", te_features.shape, te_labels.shape)

    return tr_features, tr_labels, te_features, te_labels, adj_matrix


################
# Grading Utils
################
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_model(model, optimizer):
    print(model)
    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in model.state_dict():
        print(param_tensor,"\t", model.state_dict()[param_tensor].size())
    print("optimizer's state_dict:")
    # Print optimizer's state_dict
    for var_name in optimizer.state_dict():
        print(var_name,"\t", optimizer.state_dict()[var_name])


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



def compute_ROC_AUC(test_pred, gt_labels):

    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.
    # print(gt_labels, labels_oh, test_pred.shape)
    fpr, tpr, thresh = roc_curve(labels_oh.ravel(), test_pred.ravel())
    aucroc = auc(fpr, tpr)

    return aucroc

def compute_metrics(test_pred, gt_labels):

    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.

    # print(gt_labels, labels_oh, test_pred.shape)
    # print(labels_oh, test_pred)
    idx = np.argmax(test_pred, axis=1)
    # print(gt_labels, idx)
    labels_and_pred = np.concatenate((gt_labels, idx))
    test_pred = enc.fit(labels_and_pred).transform(labels_and_pred)[gt_labels.shape[0]:, :]
    # print(test_pred)
    macro_f1_score = f1_score(labels_oh, test_pred, average='macro')
    # micro_f1_score = f1_score(labels_oh, test_pred, average='micro') #equal to accuracy.
    precision = precision_score(labels_oh, test_pred, average='macro')
    recall = recall_score(labels_oh, test_pred, average='macro')
    # kappa = cohen_kappa_score(labels_oh, test_pred)

    return macro_f1_score, precision, recall


################
# Survival Utils
################
def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # print("R mat shape:", R_mat.shape)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox



def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))



################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


