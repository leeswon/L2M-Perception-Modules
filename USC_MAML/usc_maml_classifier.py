#!/usr/bin/env python3

"""
File: usc_maml_classifier.py
Email: smr.arnold@gmail.com
Description: Trains MAML on mini-Imagenet, for the 5-ways 5-shots setting.
Usage: python usc_maml_classifier.py
"""

from abc import abstractmethod

import random
import numpy as np

import torch
from torch import nn, optim

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)


class L2MClassifier():
    @abstractmethod
    def __init__(self, model_hyperparam):
        # Initialize hyper-parameters of a lifelong learner
        raise NotImplementedError

    @abstractmethod
    def addNewTask(self, task_info, num_classes):
        # Generate/initialize task-specific sub-modules
        # task_info contains 'task_index' (enumeration of tasks) and 'task_description' (details of task)
        # num_classes is for the output size of task-specific sub-module
        raise NotImplementedError

    @abstractmethod
    def inference(self, task_info, X):
        # Make inference on the given data X according to the task (task_info)
        # return y
        raise NotImplementedError

    @abstractmethod
    def train(self, task_info, X, y):
        # Optimize trainable parameters according to the task (task_info) and data (X and y)
        raise NotImplementedError


class MAMLClassifier(L2MClassifier):
    """docstring for MAMLClassifier"""

    def __init__(self, model, fast_lr, adaptation_steps, shots, ways, device):
        super(MAMLClassifier, self).__init__()
        self.model = model
        self.maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.adaptation_steps = adaptation_steps
        self.shots = shots
        self.ways = ways
        self.device = device

    def addNewTask(self, task_info=None, num_classes=None):
        pass

    def inference(self, task_info, X=None):
        pass

    def train(self, task_info, X=None, y=None):
        learner = self.maml.clone()
        evaluation_error, evaluation_accuracy = fast_adapt(task_info,
                                                           learner,
                                                           self.loss,
                                                           self.adaptation_steps,
                                                           self.shots,
                                                           self.ways,
                                                           self.device)
        return evaluation_error, evaluation_accuracy


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        adaptation_error /= len(adaptation_data)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_error /= len(evaluation_data)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        name='mini-imagenet',
        train_samples=2*shots,
        train_ways=ways,
        test_samples=2*shots,
        test_ways=ways,
        root='~/data',
    )

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml_classifier = MAMLClassifier(
        model=model,
        fast_lr=fast_lr,
        adaptation_steps=adaptation_steps,
        shots=shots,
        ways=ways,
        device=device,
    )
    opt = optim.Adam(model.parameters(), meta_lr)

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = maml_classifier.train(task_info=batch)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()


if __name__ == '__main__':
    main()
