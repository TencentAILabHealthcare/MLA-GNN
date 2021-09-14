import os
import logging
import numpy as np
import pandas as pd
import random
import pickle

import torch

# Env
from utils import *
from model_GAT import *
from options import parse_args
from test_model import test
from model_GAT import *


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
num_splits = 15
results = []

### 2. Sets-Up Main Loop
for k in range(1, num_splits+1):
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, num_splits))
	print("*******************************************")

	tr_features, tr_labels, te_features, te_labels, adj_matrix = load_csv_data(k, opt)
	load_path = opt.model_dir + '/split' + str(k) + '_' + opt.task + '_' + str(
				opt.lin_input_dim) + 'd_all_' + str(opt.num_epochs) + 'epochs.pt'
	model_ckpt = torch.load(load_path, map_location=device)

	#### Loading Env
	model_state_dict = model_ckpt['model_state_dict']
	# hasattr(target, attr) 用于判断对象中是否含有某个属性，有则返回true.
	if hasattr(model_state_dict, '_metadata'):
		del model_state_dict._metadata

	model = GAT(opt=opt, input_dim=opt.input_dim, omic_dim=opt.omic_dim, label_dim=opt.label_dim,
				dropout=opt.dropout, alpha=opt.alpha).cuda()

	### multiple GPU
	# model = torch.nn.DataParallel(model)
	# torch.backends.cudnn.benchmark = True

	if isinstance(model, torch.nn.DataParallel): model = model.module

	print('Loading the model from %s' % load_path)
	model.load_state_dict(model_state_dict)


	### 3.2 Test the model.
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, te_features, te_fc_features = test(
		opt, model, te_features, te_labels, adj_matrix)
	GAT_te_features_labels = np.concatenate((te_features, te_fc_features, te_labels), axis=1)

	# print("model preds:", list(np.argmax(pred_test[3], axis=1)))
	# print("ground truth:", pred_test[4])
	# print(te_labels[:, 2])

	pd.DataFrame(GAT_te_features_labels).to_csv(
	    "./results/"+opt.task+"/GAT_features_"+str(opt.lin_input_dim)+"d_model/split"+str(k)+"_"+ opt.which_layer+"_GAT_te_features.csv")

	if opt.task == 'surv':
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results.append(cindex_test)
	elif opt.task == 'grad':
		print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		results.append(grad_acc_test)

	test_preds_labels = np.concatenate((pred_test[3], np.expand_dims(pred_test[4], axis=1)), axis=1)
	print(test_preds_labels.shape)
	pd.DataFrame(test_preds_labels, columns=["class1", "class2", "class3", "pred_class"]).to_csv(
		"./results/" + opt.task + "/preds/split" + str(k) + "_" + opt.which_layer + "_test_preds_labels.csv")
	# pickle.dump(pred_test, open(os.path.join(opt.results_dir, opt.task,
	# 		'preds/split%d_pred_test_%dd_%s_%depochs.pkl' % (k, opt.lin_input_dim, opt.which_layer, opt.num_epochs)), 'wb'))

print('Split Results:', results)
print("Average:", np.array(results).mean())
