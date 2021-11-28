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
          ├── split1_train_320d_features_labels.csv
          ├── split1_test_320d_features_labels.csv
          ├── ...
          ├── ...
          ├── split15_train_320d_features_labels.csv
          ├── split15_test_320d_features_labels.csv
          
      ├── input_adjacency_matrix
          ├── split1_adjacency_matrix.csv
          ├── split2_adjacency_matrix.csv
          ├── ...
          ├── ...
          ├── split15_adjacency_matrix.csv
  ```

+ For each train or test set, the RNAseq features and labels (for survival prediction and histological grading) are contained in the "xxx_xxx_320d_features_labels.csv" in the following format.
Each row represents the gene expression of one patient, while each column denotes the expression of one gene (Entrez ID):
	0	1	2	3
0	TCGA-06-0141	-0.751162972	-1.72656962	0.876216622
1	TCGA-06-0141	-0.751162972	-1.72656962	0.876216622
2	TCGA-06-0187	-0.751162972	-1.72656962	2.305385481
3	TCGA-06-0645	-0.751162972	-1.72656962	2.305385481
4	TCGA-06-0645	-0.751162972	-1.72656962	2.305385481
158	TCGA-WY-A85C	-0.751162972	0.579183132	-0.552952237
![image](https://user-images.githubusercontent.com/27730257/143766746-0dd95047-93c6-4ccf-b1db-8f6384ad0d79.png)

    0| 1 | 2 | 3 | ... | 321 | 322 | 323 
    --- | --- | --- | --- | --- | --- | --- 
    TCGA-06-0141 |-0.751162972 | -1.72656962 | 0.876216622 | ... | 1 | 313 | 2
    TCGA-06-0187 |-0.751162972 | -1.72656962 | 2.305385481 | ... | 1 | 828 | 2
    ... | ... |	... | ... |	... | ... |	... 
    TCGA-S9-A7R3 | -0.751162972 |	0.57918313 | -0.55295223 | ... | 0 | 3013 | 0
    
    ** If you want to evaluate our model with your own data, please prepare the data in this form and make sure the 
    genes are represented in the Entrez ID.


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
