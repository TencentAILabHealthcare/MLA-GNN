# MLA-GNN



This repository is an official PyTorch implementation of the paper 
**"Multi-Level Attention Graph Neural Network Based on Co-expression Gene Modules for Disease Diagnosis and Prognosis"
submitted to **Bioinformatics 2021**.

![](./figs/pipeline.png)

## Installation
### Dependencies
```
Python 3.6
PyTorch >= 1.5.0
torch_geometric
numpy
pandas
scipy
sklearn
opencv
random
```

## Data Description
For the glioma dataset, 15-fold cross validation is conducted to evaluate the model performance. For each train-test split, we compute the adjacency matrix by performing the WGCNA algorithm on the training set. The input features and adjacency matrix are saved at: ./example_data/...

The data structure is:
  ```bash
  ./example_data
  
      ├── input_features_labels                
          ├── split1_train_320_features_labels.csv
          ├── split1_test_320_features_labels.csv
          ├── ...
          ├── ...
          ├── split15_train_320_features_labels.csv
          ├── split15_test_320_features_labels.csv
          
      ├── input_adjacency_matrix
          ├── split1_adjacency_matrix.csv
          ├── split2_adjacency_matrix.csv
          ├── ...
          ├── ...
          ├── split15_adjacency_matrix.csv
  ```


## Usage

### Evaluation
```shell script
python3 test_cv.py

```

### Scripts
```shell script
test_cv.py: Load the well-trained model from the folder “/pretrained_models/…” and test the performance on the testing set of the 15 splits.

test_model.py: the definitions for "test".

model_GAT.py: the definitions for the network optimizer and the GAT network, which can be selected as 720d model(GAT_features = layer1+layer2+layer3) or 240d (either use the layer1, or layer2, or layer3 as the GAT features) model.

utils.py: contains data_loader and other functions (cindex, cox_loss, …).

options.py: Contains all the options for the argparser.

WGCNA_gbmlgg.R: compute the adjacency matrix using WGCNA method.

gradients_to_feature_importance.py: combine the gradients produced by different splits, and obtain the feature importance according to the proposed FGS mechanism.
```

## Disclaimer

This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.
