"""
Script that constructs an RNN to predict student performance.
"""
from __future__ import division

import logging
from collections import namedtuple

import numpy as np

import IRT.metrics
from IRT import metrics
from IRT.constants import (ITEM_IDX_KEY, TEMPLATE_IDX_KEY, CONCEPT_IDX_KEY, USER_IDX_KEY,
                           TIME_IDX_KEY, CORRECT_KEY, SINGLE, FIRST_COLUMN)

LOGGER = logging.getLogger(__name__)


class MLPOpts(namedtuple('MLPOpts', ['hidden_dim', 'num_iters', 'first_learning_rate', 'batch_size'])):

    def __new__(cls, hidden_dim=100, num_iters=100, first_learning_rate=0.01, batch_size=128):
        return super(MLPOpts, cls).__new__(cls, hidden_dim, num_iters, first_learning_rate, batch_size)


Results = namedtuple('Results', ['accuracy', 'auc'])


def run(data_folds, num_folds, num_questions, num_iters, data_opts, output=None, compress_dim=100,
        hidden_dim=200, test_spacing=10, recurrent=True, dropout_prob=0.0,
        output_compress_dim=None, first_learning_rate=30.0, decay_rate=0.99,
        which_fold=None):
    """ Train and test the neural net

    :param iterable data_folds: an iterator over tuples of (train, test) datasets
    :param int num_folds: number of folds total
    :param int num_questions: Total number of questions in the dataset
    :param int num_iters: Number of training iterations
    :param DataOpts data_opts: data pre-processing options. Contains the boolean `use_correct`,
        necessary for correct NN-data encoding, and all pre-processing parameters (for saving).
    :param str output: where to dump the current state of the RNN
    :param int|None compress_dim: The dimension to which to compress the data using the
        compressed sensing technique. If None, no compression is performed.
    :param int test_spacing: The number of iterations to run before running the tests
    :param int hidden_dim: The number of hidden units in the RNN.
    :param bool recurrent: Whether to use a recurrent architecture
    :param float dropout_prob: The probability of a node being dropped during training.
        Default is 0.0 (i.e., no dropout)
    :param int|None output_compress_dim: The dimension to which the output should be compressed.
        If None, no compression is performed.
    :param float first_learning_rate: The initial learning rate. Will be decayed at
        rate `decay_rate`
    :param float decay_rate: The rate of decay for the learning rate.
    :param int | None which_fold: Specify which of the folds you want to actually process. If None,
        process all folds. Good for naive parallelization.
    """
    if which_fold is not None:
        LOGGER.info("You specified to only process fold %d ", which_fold)
        if which_fold is not (1 <= which_fold <= num_folds):
            raise ValueError("which_fold ({which_fold}) must be between 1 "
                             "and num_folds({num_folds})".format(which_fold=which_fold,
                                                                 num_folds=num_folds))

    compress_dim = None if compress_dim <= 0 else compress_dim

    mlps = []
    results = []
    mlp_opts = MLPOpts(hidden_dim=hidden_dim, num_iters=num_iters, first_learning_rate=first_learning_rate)
    np.random.seed(data_opts.seed)

    for fold_num, (train_data, test_data) in enumerate(data_folds):

        fold_num += 1
        if which_fold and fold_num != which_fold:
            continue

        LOGGER.info("Beginning fold %d", fold_num)
        _, _, _, _, mlp = eval_mlp(train_data, test_data, num_questions, data_opts,
                                   mlp_opts, test_spacing, fold_num)
        mlps.append(mlp)
        results.append(mlp.results[-1])
        if output:
            with open(output + str(fold_num), 'wb') as f:
                mlps.dump(f)  # dump multiple models

    LOGGER.info("Completed all %d folds", num_folds)

    # Print overall results
    acc_sum = 0
    auc_sum = 0
    for i, result in enumerate(results):
        LOGGER.info("Fold %d Acc: %.5f AUC: %.5f", i + 1, result.accuracy, result.auc)
        acc_sum += result.accuracy
        auc_sum += result.auc

    LOGGER.info("Overall %d Acc: %.5f AUC: %.5f", i + 1, acc_sum / num_folds, auc_sum / num_folds)


def build_mlp_data(train_data):
    """
        Build data ready for MLP input
    """
    return (train_data.drop(CORRECT_KEY, axis=1).as_matrix(), train_data[CORRECT_KEY].as_matrix())


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class MLPNet(nn.Module):
    def __init__(self, num_input, num_output, hiddem_dim):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(num_input, hiddem_dim)
        self.fc2 = nn.Linear(hiddem_dim, num_output)
        LOGGER.info("MLP input_dim %d hidden_num %d output_dim %d", num_input, hiddem_dim, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))


class MLP:

    def __init__(self, train_data, opts, test_data=None, data_opts=None):

        # Setup data
        self.train_data_X = train_data[0]
        self.train_data_y = train_data[1]
        self.test_data_X = test_data[0]
        self.test_data_y = test_data[1]
        self.opts = opts
        self.data_opts = data_opts
        self.model = MLPNet(self.train_data_X.shape[1], 1, opts.hidden_dim)  # binary classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opts.first_learning_rate)
        self.loss = nn.BCELoss()  # binary cross entropy
        self.use_cuda = torch.cuda.is_available()
        self.results = []
        if self.use_cuda:
            self.model = self.model.cuda()
            LOGGER.info("Cuda detected, using cuda")

    def train_and_test(self, num_iters, test_spacing=5):
        """
        Train the RNN on the train data, and test on the test data.
        Note the learning rate is reset to self.opts.first_learning_rate
        before training commences.

        :param int num_iters: number of training iterations
        :param int test_spacing: evaluate RNN on the test data every
                                 this many training iterations.
        :return: the MAP accuracy, AYC, predicted probabilities of correct,
                 and boolean correct values on the test data.
        :rtype: float, float, np.ndarray(float), np.ndarray(float)
        """

        for epoch in np.arange(num_iters):
            self.train_data_X, self.train_data_y = unison_shuffled_copies(self.train_data_X, self.train_data_y)

            # training
            sum_loss = 0

            for instance_id in range(0, self.train_data_X.shape[0], self.opts.batch_size)[:-1]:
                train_data_X_batch = torch.from_numpy(self.train_data_X[
                                                      instance_id:min(instance_id + self.opts.batch_size,
                                                                      len(self.train_data_X))]).float()
                train_data_y_batch = torch.from_numpy(self.train_data_y[
                                                      instance_id:min(instance_id + self.opts.batch_size,
                                                                      len(self.train_data_y))])
                if self.use_cuda:
                    train_data_X_batch, train_data_y_batch = train_data_X_batch.cuda(), train_data_y_batch.cuda()

                self.optimizer.zero_grad()

                train_data_X_batch, train_data_y_batch = Variable(train_data_X_batch), Variable(train_data_y_batch)
                train_data_pred_batch = self.model(train_data_X_batch)
                loss = self.loss(train_data_pred_batch, train_data_y_batch)

                sum_loss += loss
                loss.backward()
                self.optimizer.step()

            LOGGER.info("epoch %d training loss (over all batches) %.4f ", epoch, sum_loss)
            # testing

            if (epoch % test_spacing == 0 and not epoch == 0):
                test_data_X = torch.from_numpy(self.test_data_X)
                if self.use_cuda:
                    test_data_X, test_data_y = test_data_X.cuda(), test_data_y.cuda()
                test_data_X = Variable(test_data_X, volatile=True)
                test_data_pred = self.model(test_data_X)
                test_data_pred = test_data_pred.data.numpy()

                test_acc = np.sum(test_data_pred >= 0.5)
                test_auc = metrics.auc_helper(self.test_data_y == 1, test_data_pred)
                LOGGER.info("testing every %d accuracy %.4f auc %.4f ", test_spacing, test_acc, test_auc)
                self.results.append(Results(accuracy=test_acc, auc=test_auc))

        return test_acc, test_auc, test_data_pred, test_data_pred >= 0.5


def eval_mlp(train_data, test_data, num_questions, data_opts, mlp_opts, test_spacing,
             fold_num):
    """ Create, train, and cross-validate an RNN on a train/test split.

    :param pd.DataFrame train_data: training data
    :param pd.DataFrame test_data: testing data for cross-validation (required)
    :param int num_questions: total number of questions in data
    :param DataOpts data_opts: data options
    :param RnnOpts rnn_opts: RNN options
    :param int test_spacing: test the RNN every this many iterations
    :param int fold_num: fold number (for logging and recording results only)
    :return: the trained mlp
    :rtype: MLP
    """
    LOGGER.info("Training RNN, fold %d, train length %d, test length %d", fold_num, len(train_data), len(test_data))

    train_mlp_data = build_mlp_data(train_data)
    test_mlp_data = build_mlp_data(test_data)

    mlp = MLP(train_mlp_data, mlp_opts, test_data=test_mlp_data, data_opts=data_opts)  # initialize the model

    test_acc, test_auc, test_prob_correct, test_corrects = mlp.train_and_test(
        mlp_opts.num_iters,
        test_spacing=test_spacing)  # AUC score for the case is 0.5. A score for a perfect classifier would be 1.
    LOGGER.info("Fold %d: Num Interactions: %d; Test Accuracy: %.5f; Test AUC: %.5f",
                fold_num, len(test_data), test_acc, test_auc)

    return test_acc, test_auc, test_prob_correct, test_corrects, mlp
