import os
import sys
import logging
import numpy as np
import random
import pickle

import torch
import pandas as pd

# Env
from utils import *
from options import parse_args
from train_test_new import train, test

from load_CRCSC_data import load_features_labels, CRC_Dataset, construct_graph

### 1. Initializes parser and device
opt = parse_args()

log_path = "./logs/"
snapshot_path = opt.model_dir + opt.exp + "/"


if __name__ == "__main__":

    logging.basicConfig(filename=log_path + opt.exp +".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(opt))

    tr_features, norm_tr_features, te_features, tr_labels, te_labels, keep_idx, all_test_cohorts, \
            tr_sample_ids, te_sample_ids = load_features_labels(retain_dim=opt.num_nodes)
    adj_matrix, edge_index = construct_graph(opt, norm_tr_features, keep_idx)
    print("train data:", tr_features.shape, tr_labels.shape)
    print("test data:", te_features.shape, te_labels.shape)

    train_dataset = CRC_Dataset(feature=tr_features, label=tr_labels, edge=edge_index)
    test_dataset = CRC_Dataset(feature=te_features, label=te_labels, edge=edge_index)

    print("====================dataset loaded====================")

    ### 2.1 Train Model
    # model, optimizer, metric_logger = train(opt, train_dataset, test_dataset)
    model, optimizer, metric_logger, class_edge_weights, class_node_importance, \
            overall_edge_weights, overall_node_importance = train(opt, train_dataset, test_dataset)

    ### 2.2 Test Model
    loss_train, grad_acc_train, pred_train, tr_features, tr_fc_features = test(opt, model, train_dataset)
    loss_test, grad_acc_test, pred_test, te_features, te_fc_features = test(opt, model, test_dataset)
    # test_probs = np.exp(pred_test)/np.sum(np.exp(pred_test), axis=1)
    # print("test GAT features:", te_features.shape)
    # print("test fc features:", te_fc_features.shape)
    # print("test preds:", test_probs)
    all_metrics = compute_cohort_metrics(pred_test[0], np.uint(pred_test[1]), all_test_cohorts)
    print(all_metrics)
    test_results = {'sample_id': te_sample_ids, 'GNN_pred': np.argmax(pred_test[0], axis=1),
                    'CMS_network': np.uint(pred_test[1])}
    # pd.DataFrame(test_results).to_csv('./results/GNN_HumanNet_preds.csv')
    # pd.DataFrame(test_results).to_csv('./results/GNN_sim_graph_preds.csv')
    # pd.DataFrame(test_results).to_csv('./GNN_sim_top_genes_predictions/top' + str(opt.num_nodes) + '_test.csv')
    #
    # train_results = {'GNN_pred': np.argmax(pred_test[0], axis=1), 'label': np.uint(pred_test[1])}
    # pd.DataFrame(train_results).to_csv('./GNN_sim_top_genes_predictions/top' + str(opt.num_nodes) + '_train.csv')

    # edge_weights_file = opt.results_dir + "GNN_sim_graph_edge_weights_wo_elu/"
    # feat_importance_file = opt.results_dir + "GNN_sim_graph_feature_importance_wo_elu/"
    # # print(edge_weights_file, feat_importance_file)
    # 
    # # edge_weights_file = opt.results_dir + "GNN_HumanNet_edge_weights/"
    # # feat_importance_file = opt.results_dir + "GNN_HumanNet_feature_importance/"
    # 
    # for i in range(opt.label_dim):
    #     # print(class_edge_weights[i])
    #     pd.DataFrame(class_edge_weights[i]).to_csv(edge_weights_file + "class" + str(i) + ".csv")
    #     pd.DataFrame(class_node_importance[i]).to_csv(feat_importance_file + "class" + str(i) + ".csv")
    # 
    # pd.DataFrame(overall_edge_weights).to_csv(edge_weights_file + "overall.csv")
    # pd.DataFrame(overall_node_importance).to_csv(feat_importance_file + "overall.csv")

    print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
    print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
    logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))


    ### 2.3 Saves Model
    model_state_dict = model.state_dict()
    save_path = opt.model_dir + opt.exp + '.pt'
    print("Saving model at:", save_path)

    torch.save({
        'opt': opt,
        'epoch': opt.num_epochs,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        save_path)


    # pickle.dump(pred_train, open(os.path.join(opt.results_dir,
    #     'preds/pred_train_%s_%depochs.pkl' % (opt.which_layer, opt.num_epochs)), 'wb'))
    # pickle.dump(pred_test, open(os.path.join(opt.results_dir,
    #     'preds/pred_test_%s_%depochs.pkl' % (opt.which_layer, opt.num_epochs)), 'wb'))
