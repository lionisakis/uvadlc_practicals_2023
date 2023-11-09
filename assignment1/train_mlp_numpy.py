################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_classes=np.unique(targets)
    conf_mat=np.zeros((total_classes,total_classes))
    for i in range(total_classes):
      for j in range(total_classes):
        predicted=np.sum(np.where(predictions==i),1,0)
        truth=np.sum(np.where(targets==j),1,0)
        conf_mat[i,j]= predicted+truth
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    metrics={}
    total_classes=confusion_matrix.shape()[0]
    TP,FP,FN,TN=[],[],[],[]
    for i in range(total_classes):
      TP.append(confusion_matrix[i,i])
      FP.append(confusion_matrix[i,:i]+confusion_matrix[i,i:])
      FN.append(confusion_matrix[:i,i]+confusion_matrix[i:,i])
      tmp=0
      for j in range(total_classes):
        if j==i:
          continue
        tmp+=confusion_matrix[:i,i]
      TN.append(tmp)
    TP,FP,FN,TN=np.array(TP),np.array(FP),np.array(FN),np.array(TN)
    metrics["accuracy"]=np.sum(TP+FP)/np.sum(TP+FP+FN+TN)
    metrics["precision"]=np.sum(TP,axis=0)/np.sum(TP+FP,axis=0)
    metrics["recall"]=np.sum(TP,axis=0)/np.sum(TP+FN,axis=0)
    metrics["f1_beta"]=((1+np.power(beta,2))*metrics["precision"]*metrics["recall"])/(np.power(beta,2)*metrics["precision"]+metrics["recall"])
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy=[]
    precision=[]
    recall=[]
    f1_beta=[]
    for x,y in data_loader:
      predictions=model.forward(x)
      confusion_matrix=confusion_matrix(predictions,y)
      tmp=confusion_matrix_to_metrics(confusion_matrix, beta=1.)
      accuracy.append(tmp["accuracy"])
      precision.append(tmp["precision"])
      recall.append(tmp["recall"])
      f1_beta.append(tmp["f1_beta"])
    
    metrics={}
    metrics["accuracy"]=np.mean(np.array(accuracy))
    metrics["precision"]=precision
    metrics["recall"]=recall
    metrics["f1_beta"]=f1_beta    
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """
    logging_dict={}
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(hidden_dims, lr, batch_size, epochs, seed, data_dir)
    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    print(cifar10_loader["train"])
    # print('cifar10_loader["train"].shape',cifar10_loader["train"].x.shape)
    model = MLP(3072,hidden_dims, 10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies=[]
    for epoch in range(epochs):
      model.forward(cifar10_loader["train"])
      loss_module.forward(model.output,cifar10_loader["train"])
      model.backward(loss_module.backward())
      for w in model.weights:
        w.weight=w.weight-lr*w.grad_weight
        w.bias=w.bias-lr*w.grad_bias
      val_accuracies.append(evaluate_model(model, cifar10_loader["validation"], num_classes=10))
    # TODO: Test best model
    test_accuracy = evaluate_model(model, cifar10_loader["test"], num_classes=10)
    # TODO: Add any information you might want to save for plotting
    logging_info = [epoch]
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


# if __name__ == '__main__':
#     # Command line arguments
#     parser = argparse.ArgumentParser()
    
#     # Model hyperparameters
#     parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
#                         help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
#     # Optimizer hyperparameters
#     parser.add_argument('--lr', default=0.1, type=float,
#                         help='Learning rate to use')
#     parser.add_argument('--batch_size', default=128, type=int,
#                         help='Minibatch size')

#     # Other hyperparameters
#     parser.add_argument('--epochs', default=10, type=int,
#                         help='Max number of epochs')
#     parser.add_argument('--seed', default=42, type=int,
#                         help='Seed to use for reproducing results')
#     parser.add_argument('--data_dir', default='data/', type=str,
#                         help='Data directory where to store/find the CIFAR10 dataset.')

#     args = parser.parse_args()
#     kwargs = vars(args)
#     print(kwargs)
#     model, val_accuracies, test_accuracy, logging_dict=train(**kwargs)
#     # Feel free to add any additional functions, such as plotting of the loss curve here
#     matplotlib.pyplot.plot(val_accuracies["accuracy"],label="Validation Accuracy")
#     matplotlib.pyplot.legend()
#     matplotlib.pyplot.savefig(args[-1]+"/accuracy.png")
