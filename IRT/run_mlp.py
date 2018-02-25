"""
Script that constructs an RNN to predict student performance.
"""
from __future__ import division

import logging
from collections import namedtuple
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import IRT.metrics
from IRT import metrics
from IRT.constants import (ITEM_IDX_KEY, TEMPLATE_IDX_KEY, CONCEPT_IDX_KEY, USER_IDX_KEY,
                           TIME_IDX_KEY, CORRECT_KEY, SINGLE, ORDER_ID, USER_ID_KEY, ITEM_ID_KEY)

LOGGER = logging.getLogger(__name__)


class MLPOpts(namedtuple('MLPOpts', ['hidden_dim', 'num_iters', 'first_learning_rate', 'batch_size', 'embedding_dim'])):

    def __new__(cls, hidden_dim=100, num_iters=100, first_learning_rate=0.001, batch_size=128, embedding_dim=200):
        return super(MLPOpts, cls).__new__(cls, hidden_dim, num_iters, first_learning_rate, batch_size, embedding_dim)


Results = namedtuple('Results', ['accuracy', 'auc'])


def run(data_folds, num_folds, num_questions, num_iters, data_opts,
        hidden_dim=200, test_spacing=10,
        first_learning_rate=0.001, which_fold=None, num_users=None, user_ids=None, item_ids=None, embedding_dim=200):
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

    mlps = []
    results = []
    mlp_opts = MLPOpts(hidden_dim=hidden_dim, num_iters=num_iters, first_learning_rate=first_learning_rate,
                       embedding_dim=embedding_dim)
    np.random.seed(data_opts.seed)

    for fold_num, (train_data, test_data) in enumerate(data_folds):

        LOGGER.info("this fold has " + str(len(test_data[TIME_IDX_KEY].unique())) + " interactions and students " + str(
            len(test_data[USER_IDX_KEY].unique())))

        fold_num += 1
        if which_fold and fold_num != which_fold:
            continue

        LOGGER.info("Beginning fold %d", fold_num)
        _, _, _, _, mlp = eval(train_data, test_data, num_questions, data_opts,
                               mlp_opts, test_spacing, fold_num, num_users, user_ids, item_ids)

        mlps.append(mlp)
        results.append(mlp.results[-1])
        # if output:
        #     with open(output + str(fold_num), 'wb') as f:
        #         mlps.dump(f)  # dump multiple models

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
    # input(train_data[CORRECT_KEY])
    train_data_label = train_data[[CORRECT_KEY]]
    train_data_label.loc[:, "in" + CORRECT_KEY] = 1 - train_data_label.loc[:, CORRECT_KEY]
    # input(train_data_label[["in" + CORRECT_KEY, CORRECT_KEY]].as_matrix())

    # return (train_data.drop(CORRECT_KEY, axis=1).as_matrix(), train_data[CORRECT_KEY].as_matrix())

    # input(train_data.drop([CORRECT_KEY, USER_IDX_KEY,ITEM_IDX_KEY,TIME_IDX_KEY],axis=1))
    return (
        train_data.drop([CORRECT_KEY, USER_IDX_KEY, ITEM_IDX_KEY, TIME_IDX_KEY], axis=1).as_matrix(),
        train_data_label[["in" + CORRECT_KEY, CORRECT_KEY]].as_matrix())


def build_ncf_data(train_data, num_users, num_questions, user_ids, item_ids):
    """
            Build data ready for NCF input
    """

    input("train_data in build_ncf_data")
    input(train_data)
    train_data_label = train_data[[CORRECT_KEY]]
    train_data_label.loc[:, "in" + CORRECT_KEY] = 1 - train_data_label.loc[:, CORRECT_KEY]
    a = train_data[USER_IDX_KEY].unique()
    b = train_data[ITEM_IDX_KEY].unique()
    # input("user id onehot encode? three numbers hould be the same "+str(max(a))+" "+str(num_users)+" "+str(len(user_ids)))
    # input("3 same "+" "+str(max(b))+" "+str(num_questions)+" "+str(len(item_ids)))
    # input(train_data[[USER_IDX_KEY,ITEM_IDX_KEY,TIME_IDX_KEY]].as_matrix())

    return (
        train_data[[USER_IDX_KEY, ITEM_IDX_KEY, TIME_IDX_KEY]].as_matrix(),
        train_data_label[["in" + CORRECT_KEY, CORRECT_KEY]].as_matrix(), train_data[USER_ID_KEY],
        train_data[ITEM_ID_KEY], train_data[ORDER_ID])


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class MLP(nn.Module):
    def __init__(self, num_input, num_output, hiddem_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_input, hiddem_dim)
        self.fc2 = nn.Linear(hiddem_dim, num_output)
        LOGGER.info("MLP input_dim %d hidden_num %d output_dim %d", num_input, hiddem_dim, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # return F.sigmoid(self.fc2(x))
        return F.softmax(self.fc2(x), dim=1)


class NCF(nn.Module):
    def __init__(self, num_input, num_output, hiddem_dim, num_users, num_questions, emb_size):
        super(NCF, self).__init__()
        LOGGER.info("Building model: embedding size " + str(emb_size) + " hidden dimension " + str(hiddem_dim))

        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_questions, emb_size)
        self.fc1 = nn.Linear(2 * emb_size + 1, hiddem_dim)
        self.fc2 = nn.Linear(hiddem_dim, num_output)
        LOGGER.info("NCF input_dim=2*emb_size+1(timestamp) %d hidden_num %d output_dim %d", 2 * emb_size + 1,
                    hiddem_dim, num_output)

    def forward(self, words):
        user_emb = self.user_embedding(words[:, 0].long())  # 2D Tensor of size [batch_size x emb_size]
        item_emb = self.item_embedding(words[:, 1].long())
        # input("user emb shape "+str(user_emb.shape)+" "+str(type(user_emb))+" is float Tensor?")
        # input("item emb shape "+str(item_emb.shape))
        batch_size = words.shape[0]
        # input(batch_size)
        # input("timestamp feat shape "+str(words[:,2].contiguous().view(batch_size,1).shape))
        x = torch.cat([user_emb, item_emb, words[:, 2].contiguous().view(batch_size, 1)], dim=1)
        # input("concatenated user item embedding + timestamp shape "+str(x.shape)+" type "+str(type(x.data)))
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ModelExecuter:

    def __init__(self, train_data, opts, test_data=None, data_opts=None, num_users=None, num_questions=None):

        use_mlp = (num_users == None) and (num_questions == None)

        # Setup data
        self.train_data_X = train_data[0]
        self.train_data_y = np.asarray(train_data[1], dtype=np.int32)

        self.test_data_X = test_data[0]
        self.test_data_y = test_data[1]  # shape is #instance * 2, where 2 are 2 columns representing incorrect, correct

        if not use_mlp:
            self.test_data_user_ids = test_data[2]
            self.test_data_item_ids = test_data[3]
            self.test_data_order_ids = test_data[4]

        self.opts = opts
        self.data_opts = data_opts
        if use_mlp:
            self.model = MLP(self.train_data_X.shape[1], 2, opts.hidden_dim)  # binary classification
        else:
            self.model = NCF(self.train_data_X.shape[1], 2, opts.hidden_dim, num_users, num_questions,
                             opts.embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opts.first_learning_rate)
        LOGGER.info("Model optimizer first learning rate is " + str(opts.first_learning_rate))
        # self.loss = nn.BCELoss()  # binary cross entropy
        self.loss = nn.MSELoss()
        self.use_cuda = torch.cuda.is_available()
        self.results = []
        if self.use_cuda:
            self.model = self.model.cuda()
            LOGGER.info("Cuda detected, using cuda")
        self.prediction_output = None
        if data_opts.prediction_output != None:
            self.prediction_output = data_opts.prediction_output
            LOGGER.info("Prediction output path " + self.prediction_output)

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
        LOGGER.info("Number of iterations for this fold " + str(num_iters))

        for epoch in np.arange(num_iters):
            LOGGER.info("Epoch " + str(epoch) + " starts! ")
            self.train_data_X, self.train_data_y = unison_shuffled_copies(self.train_data_X, self.train_data_y)

            # training
            sum_loss = 0

            for instance_id in range(0, self.train_data_X.shape[0], self.opts.batch_size)[:-1]:
                train_data_X_batch = torch.from_numpy(self.train_data_X[
                                                      instance_id:min(instance_id + self.opts.batch_size,
                                                                      self.train_data_X.shape[0])]).float()

                # input(train_data_X_batch)
                # input(self.train_data_X.shape)
                # input(self.train_data_X)
                #
                # input(self.train_data_y.shape)
                # input(self.train_data_y.dtype)
                train_data_y_batch = torch.from_numpy(self.train_data_y[
                                                      instance_id:min(instance_id + self.opts.batch_size,
                                                                      self.train_data_y.shape[0])]).float()

                # LOGGER.info("current batch positive "+str(1.0*np.sum(self.train_data_y[
                #                                       instance_id:min(instance_id + self.opts.batch_size,
                #                                                       len(self.train_data_y))]==1)/len(self.train_data_y[
                #                                       instance_id:min(instance_id + self.opts.batch_size,
                #                                                       len(self.train_data_y))])))

                # input(train_data_y_batch)
                if self.use_cuda:
                    train_data_X_batch, train_data_y_batch = train_data_X_batch.cuda(), train_data_y_batch.cuda()

                self.optimizer.zero_grad()

                train_data_X_batch, train_data_y_batch = Variable(train_data_X_batch), Variable(train_data_y_batch,
                                                                                                requires_grad=False)

                # input("target "+str(train_data_y_batch.data))
                train_data_pred_batch = self.model(train_data_X_batch)
                # input("prediction "+str(train_data_pred_batch.data))
                loss = self.loss(train_data_pred_batch, train_data_y_batch)

                sum_loss += loss
                loss.backward()
                self.optimizer.step()

            LOGGER.info("epoch %d training loss (over all batches) %.4f ", epoch, sum_loss)
            # testing

            if (epoch % test_spacing == 0 and not epoch == 0):
                test_data_X = torch.from_numpy(self.test_data_X).float()
                if self.use_cuda:
                    test_data_X = test_data_X.cuda()
                test_data_X = Variable(test_data_X, volatile=True)
                test_data_pred = self.model(test_data_X).data
                if self.use_cuda:
                    test_data_pred = test_data_pred.cpu()
                test_data_pred = test_data_pred.numpy()

                LOGGER.info("self.test_data_y.shape " + str(self.test_data_y.shape))
                LOGGER.info("planning to write to csv " + self.prediction_output)
                # LOGGER.info(str(test_data_pred[:20]))

                # test_acc = np.sum(test_data_pred[:, 1] >= test_data_pred[:, 0]) / test_data_pred.shape[0]
                test_acc = accuracy_score(self.test_data_y[:, 1],
                                          np.asarray((test_data_pred[:, 1] >= test_data_pred[:, 0]), dtype=int))

                # input(self.test_data_y[:, 1].shape)
                # input(test_data_pred[:, 1])
                fpr, tpr, thresholds = roc_curve(self.test_data_y[:, 1], test_data_pred[:, 1], pos_label=1)
                test_auc = auc(fpr, tpr)

                # test_auc = metrics.auc_helper(self.test_data_y[:, 1] == 1, test_data_pred[:, 1])
                LOGGER.info("testing every %d accuracy %.4f auc %.4f ", test_spacing, test_acc, test_auc)
                self.results.append(Results(accuracy=test_acc, auc=test_auc))

        df=pd.DataFrame()
        df[ORDER_ID]=self.test_data_order_ids
        df[USER_ID_KEY]=self.test_data_user_ids
        df[ITEM_ID_KEY]=self.test_data_item_ids
        df[CORRECT_KEY]=self.test_data_y[:,1]
        df["prediction"]=test_data_pred[:,1]

        if os._exists(self.prediction_output):
            with open(self.prediction_output, 'a') as f:
                LOGGER.info("Appending this fold prediction to "+self.prediction_output+" number of interactions "+str(len(test_data_pred[:,1])))
                df.to_csv(f, header=False)
        else:
            with open(self.prediction_output, 'w') as f:
                LOGGER.info(
                    "Creating 1st fold prediction csv to " + self.prediction_output + " number of interactions " + str(
                        len(test_data_pred[:, 1])))
                df.to_csv(f)

        return test_acc, test_auc, test_data_pred[:, 1], test_data_pred[:, 1] >= test_data_pred[:, 0]


def eval(train_data, test_data, num_questions, data_opts, mlp_opts, test_spacing,
         fold_num, num_users, user_ids, item_ids):
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
    LOGGER.info("num_questions=26684? " + str(num_questions) + " num_users=4097? " + str(num_users))
    if data_opts.meta:
        train_mlp_data = build_mlp_data(train_data)
        test_mlp_data = build_mlp_data(test_data)

        model = ModelExecuter(train_mlp_data, mlp_opts, test_data=test_mlp_data,
                              data_opts=data_opts)  # initialize the model
    else:
        # input("user id example " + str(user_ids[5]) + " item_ids example " + str(item_ids[5])) # original id value
        train_ncf_data = build_ncf_data(train_data, num_users, num_questions, user_ids, item_ids)
        test_ncf_data = build_ncf_data(test_data, num_users, num_questions, user_ids, item_ids)
        model = ModelExecuter(train_ncf_data, mlp_opts, test_data=test_ncf_data, data_opts=data_opts,
                              num_users=num_users,
                              num_questions=num_questions)  # initialize the model

    test_acc, test_auc, test_prob_correct, test_corrects = model.train_and_test(
        mlp_opts.num_iters,
        test_spacing=test_spacing)  # AUC score for the case is 0.5. A score for a perfect classifier would be 1.

    LOGGER.info("Fold %d: Num Interactions: %d; Test Accuracy: %.5f; Test AUC: %.5f",
                fold_num, len(test_data), test_acc, test_auc)

    return test_acc, test_auc, test_prob_correct, test_corrects, model
