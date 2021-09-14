"""
This script converts the feature gradients to the importance score of each node in each layer.
Input: gradients, 3 layers * 240 nodes = 720 features in total. gradients.shape = [n, 720].
Output: the feature importance scores of each layer, 240 rows (nodes) * 4columns (classes).
"""

import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd


def gradients_each_class(gradients, labels):
    """
    according to the labels, separate the gradients for each class.
    return: all_class_importance: [3, 240, 3], the first dimension is 3 classes(class0, class1, class2, overall),
    the second dimension is 240 nodes, the last dimension is 3 GAT layers.
    """
    class0_index = np.argwhere(labels == 0)
    class0_gradients = gradients[class0_index[:, 0], :]
    class0_importance = np.mean(class0_gradients, axis=0)
    class0_importance = np.reshape(class0_importance, (3, 240)).T
    class0_importance = np.expand_dims(normalize(class0_importance, axis=0, norm='max'), axis=0)

    class1_index = np.argwhere(labels == 1)
    class1_gradients = gradients[class1_index[:, 0], :]
    class1_importance = np.mean(class1_gradients, axis=0)
    class1_importance = np.reshape(class1_importance, (3, 240)).T
    class1_importance = np.expand_dims(normalize(class1_importance, axis=0, norm='max'), axis=0)

    class2_index = np.argwhere(labels == 2)
    class2_gradients = gradients[class2_index[:, 0], :]
    class2_importance = np.mean(class2_gradients, axis=0)
    class2_importance = np.reshape(class2_importance, (3, 240)).T
    class2_importance = np.expand_dims(normalize(class2_importance, axis=0, norm='max'), axis=0)

    # print(class0_gradients.shape, class1_gradients.shape, class2_gradients.shape)
    # print(class0_importance.shape, class1_importance.shape, class2_importance.shape)

    overall_importance = (class0_importance+class1_importance+class2_importance)/3.0
    all_class_importance = np.concatenate((
        class0_importance, class1_importance, class2_importance, overall_importance), axis=0) # [4, 240, 3]
    # print(all_class_importance.shape)

    return all_class_importance




def all_class_feature_importance():
    """
    load the feature gradients, and then separate the gradients of the samples from each class,
    and calculate the average gradients as the feature importance of each class.
    """

    feature_importance_all = np.zeros((4, 240, 3))

    for k in range(1, 16):
        gradients_labels = np.array(pd.read_csv(
            './results/grad/feature_gradients/split' + str(k) + '_720d.csv')).astype(float)[:, 1:]
        gradients = gradients_labels[:, :720]
        labels = gradients_labels[:, -1]

        all_class_importance = gradients_each_class(gradients, labels)# [4, 240, 3]
        ### average the feature importance from three GAT layers.
        # all_class_importance = np.mean(all_class_importance, axis=1, keepdims=True)
        feature_importance_all += all_class_importance # [4, 240, 3]

    avg_layer_importance = np.mean(feature_importance_all, axis=-1, keepdims=True)
    feature_importance_all = np.concatenate((feature_importance_all, avg_layer_importance), axis=-1) # [4,240,4]
    print(feature_importance_all.shape)

    header = np.array(['class0', 'class1', 'class2', 'all_classes']).reshape(1, 4)
    layers = ['layer1', 'layer2', 'layer3', 'overall']

    for layer in range(4):
        layer_importance = feature_importance_all[:, :, layer].T # [240, 4]
        layer_importance = np.concatenate((header, layer_importance), axis=0)
        # print(layer_importance)
        pd.DataFrame(layer_importance).to_csv(
             "./results/grad/feature_gradients/" + layers[layer] + "_feature_importance.csv", header=0, index=0)


all_class_feature_importance()
