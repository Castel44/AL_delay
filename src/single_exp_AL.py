import os
import argparse
import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import shutil
import itertools
import time

from datetime import datetime
from collections import deque

import multiprocessing as mp

import pandas as pd
from skactiveml.utils import call_func
from skactiveml.classifier import PWC
from skactiveml.stream import FixedUncertainty, VariableUncertainty, Split, PALS, RandomSampler, PeriodicSampler
from skactiveml.stream.budget_manager import FixedThresholdBudget, FixedUncertaintyBudget, BIQF, \
    VariableUncertaintyBudget, SplitBudget, EstimatedBudget
from skactiveml.stream.verification_latency import BaggingDelaySimulationWrapper, ForgettingWrapper, \
    FuzzyDelaySimulationWrapper
from skactiveml.classifier import SklearnClassifier

from skmultiflow.drift_detection import PageHinkley, KSWIN, EDDM, DDM, HDDM_W, HDDM_A
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.bayes import NaiveBayes

from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier

from src.concept_drift.skflow_detectors import HDDDM

from src.utils.saver import Saver
from src.utils.new_dataload import OUTPATH, DATAPATH
from src.KDE_uncert.drifting_dataset import create_dataset
from src.KDE_uncert.propagate_wrapper import PropagateLabelDelayWrapper

from scipy.spatial.distance import cdist


class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def batch_latency(length, delay):
    if delay == 0:
        return np.full(length, 0)
    o = []
    for i in range(length):
        o.append(-(i % delay) + delay)
    return np.asarray(o)


def propagate_label(X, y, acquisitions, k=3):
    y_type = y.dtype
    new_y = np.copy(y)

    queried_delay = np.intersect1d(np.where(acquisitions)[0], np.where(np.isnan(y))[0])
    available_labels = np.where(~np.isnan(y))[0]

    k = np.min([k, len(available_labels) - 1])

    pairwise_distance = cdist(X[queried_delay], X[available_labels])
    knn_idx = np.argpartition(pairwise_distance, k, axis=1)[:, :k]  # idx of closest neighbors reference data

    for j, queried_idx in enumerate(queried_delay):
        knn_labels = y[available_labels][knn_idx][j].astype(int)
        unique_labels = np.unique(knn_labels)
        # TODO: normalize weight to add. Might incurr in problems
        # w_t = ty[knn_idx][j] # time weight. Higher is better
        # w_d = 1 / (pairwise_distance[j][knn_idx][j] + 1e-10)# distance weight.
        weighted_count = np.bincount(knn_labels, weights=None)
        l_ = knn_labels[np.argmax(unique_labels)].astype(y_type)
        new_y[queried_idx] = l_
    return new_y


def propagate(X, y, ty, k=5, weight=True):
    n = np.sum(~np.isnan(y))
    available_labels = np.where(~np.isnan(y))[0]

    different_labels = len(np.unique(y[available_labels]))
    y_nan = np.where(np.isnan(y))[0]
    k = np.min([k, len(available_labels) - 1])
    y_type = y.dtype
    new_y = np.copy(y)

    if different_labels > 1 and len(y_nan) > 0:
        pairwise_distance = cdist(X[y_nan], X[available_labels])
        knn_idx = np.argpartition(pairwise_distance, k, axis=1)[:,:k]  # idx of closest neighbors reference data
        # d_max = pairwise_distance.max()
        # d_min = pairwise_distance.min()
        # normalized_distance = (pairwise_distance - d_min) / (d_max - d_min)
        normalized_ty = ty / ty.max()

        for j, idx in enumerate(y_nan):
            knn_labels = y[available_labels][knn_idx][j].astype(int)
            # unique_labels = np.unique(knn_labels)
            w = None
            if weight:
                w = normalized_ty[knn_idx][j]  # time weight. Higher is better

            weighted_count = np.bincount(knn_labels, weights=w)
            l_ = np.argmax(weighted_count).astype(y_type)
            new_y[idx] = l_
    return new_y


def single_experiment(args, path):
    path = os.path.join(path, 'seed_{}'.format(args.init_seed))
    np.random.seed(args.init_seed)
    random_state = np.random.RandomState(args.init_seed)

    # Suppress output
    if args.disable_print:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    saver = SaverSlave(path)
    df = pd.DataFrame()

    ######################################################################################################
    X, y, drifts = create_dataset(args.dataset, 2, args.n_classes, args.stream_length, args.n_concept_drifts)

    stream_length = len(X)
    init_train_length = args.init_train_length
    n_classes = len(np.unique(y))
    training_size = args.training_size
    budget = args.budget

    ######################################################################################################
    # Define methods and strategies

    clf_factory = {
        'PWC': lambda: PWC(classes=[x for x in range(n_classes)], random_state=get_randomseed(random_state)),
        'SVC': lambda: SklearnClassifier(SVC(probability=True), classes=[x for x in range(n_classes)],
                                         random_state=get_randomseed(random_state)),
        'HT': lambda: SklearnClassifier(HoeffdingTreeClassifier(), classes=[x for x in range(n_classes)],
                                        random_state=get_randomseed(random_state)),
        'KNN': lambda: SklearnClassifier(KNNClassifier(n_neighbors=5, max_window_size=training_size),
                                         classes=[x for x in range(n_classes)],
                                         random_state=get_randomseed(random_state)),
        'NB': lambda: SklearnClassifier(NaiveBayes(), classes=[x for x in range(n_classes)],
                                        random_state=get_randomseed(random_state)),
    }

    clf_model = clf_factory[args.model]
    missing_label = clf_model().missing_label

    query_strategies_factories = {
        'RandomSampler': lambda: RandomSampler(random_state=get_randomseed(random_state),
                                               budget_manager=FixedThresholdBudget(budget=budget)),
        'PeriodicSampler': lambda: PeriodicSampler(random_state=get_randomseed(random_state),
                                                   budget_manager=FixedThresholdBudget(budget=budget)),
        'FixedUncertainty': lambda: FixedUncertainty(random_state=get_randomseed(random_state),
                                                     budget_manager=FixedUncertaintyBudget(w=256,
                                                                                           budget=budget)),
        'VariableUncertainty': lambda: VariableUncertainty(random_state=get_randomseed(random_state),
                                                           budget_manager=VariableUncertaintyBudget(w=256,
                                                                                                    budget=budget)),
        'Split': lambda: Split(random_state=get_randomseed(random_state),
                               budget_manager=SplitBudget(w=256, budget=budget)),
        'PALS': lambda: PALS(random_state=get_randomseed(random_state),
                             budget_manager=BIQF(w=256, w_tol=int(200*budget), budget=budget))
    }

    delay_wrappers_factories = {
        'None': lambda qs: qs,
        'Forgetting': lambda qs: ForgettingWrapper(base_query_strategy=qs, w_train=training_size,
                                                   random_state=get_randomseed(random_state)),
        'Propagate': lambda qs: PropagateLabelDelayWrapper(base_query_strategy=qs,
                                                           random_state=get_randomseed(random_state),
                                                           k=args.k_neighbours, weighted=args.weighted),
        'Bagging': lambda qs: BaggingDelaySimulationWrapper(base_query_strategy=qs,
                                                            random_state=get_randomseed(random_state), K=args.K,
                                                            delay_prior=args.delay_prior),
        'Fuzzy': lambda qs: FuzzyDelaySimulationWrapper(base_query_strategy=qs,
                                                        random_state=get_randomseed(random_state),
                                                        delay_prior=args.delay_prior)
    }
    delay_wrappers_factories["Forgetting + Bagging"] = lambda qs: delay_wrappers_factories["Forgetting"](
        delay_wrappers_factories["Bagging"](qs))
    delay_wrappers_factories["Forgetting + Fuzzy"] = lambda qs: delay_wrappers_factories["Forgetting"](
        delay_wrappers_factories["Fuzzy"](qs))
    delay_wrappers_factories["Forgetting + Propagate"] = lambda qs: delay_wrappers_factories["Forgetting"](
        delay_wrappers_factories["Propagate"](qs))

    ######################################################################################################
    verification_latency_pool = args.verification_latency
    delta_latency_pool = args.delta_latency
    query_pool = args.query_strategy
    delay_pool = args.delay_strategy
    tot_iter = len(verification_latency_pool) * len(delta_latency_pool) * len(query_pool) * len(delay_pool)

    n = 0
    start_time = time.time()
    for verification_latency in verification_latency_pool:
        for delta_latency in delta_latency_pool:
            # Prepare data
            X_init = X[:init_train_length, :]
            y_init = y[:init_train_length]
            X_stream = X[init_train_length:, :]
            y_stream = y[init_train_length:]

            # sample arrival
            tX = np.arange(stream_length)

            if args.latency_type == 'fixed':
                latency = verification_latency
                if delta_latency == 0:
                    latency_hat = verification_latency
                else:
                    latency_hat = delta_latency

            elif args.latency_type == 'batch':
                latency = batch_latency(stream_length, verification_latency)
                if delta_latency == 0:
                    latency_hat = latency
                else:
                    latency_hat = delta_latency

            elif args.latency_type == 'random':
                latency = np.random.uniform(0, verification_latency, stream_length).astype(int)
                # if delta_latency == 0 and verification_latency != 0:
                #    print(f'latency_type={args.latency_type}: latency_hat can not be zero. Skipping..')
                #    continue
                if delta_latency == 0:
                    latency_hat = latency
                else:
                    latency_hat = delta_latency

            else:
                raise ValueError
            print(f'latency type:{args.latency_type}, {verification_latency=}, {latency_hat=}')

            # label arrival
            ty = tX + latency
            tX_init = tX[:init_train_length]
            ty_init = tX[:init_train_length]
            tX_stream = tX[init_train_length:]
            ty_stream = ty[init_train_length:]
            ty_ = tX + latency_hat
            ty_init_ = tX[:init_train_length]
            ty_stream_ = ty_[init_train_length:]

            #
            query_strategies = args.query_strategy
            delay_strategies = args.delay_strategy

            for query_strategy_name in query_strategies:
                query_strategy = query_strategies_factories[query_strategy_name]
                if args.verbose > 0:
                    print("Query Strategy: ", query_strategy_name)
                fig, ax = plt.subplots()

                for delay_strategy_name in delay_strategies:
                    iter_start_time = time.time()
                    delay_strategy = delay_wrappers_factories[delay_strategy_name]
                    # print("Delay Strategy: ", delay_strategy_name)

                    clf = clf_model()
                    # initializing the query strategy
                    delay_wrapper = delay_strategy(query_strategy())
                    # initializing the training data
                    X_train = deque(maxlen=training_size)
                    X_train.extend(X_init)
                    y_train = deque(maxlen=training_size)
                    y_train.extend(y_init)
                    # y_train_nonan = deque(maxlen=training_size)
                    # y_train_nonan.extend(y_init)
                    # X_train_nonan = deque(maxlen=training_size)
                    # X_train_nonan.extend(X_init)
                    # initialize the time stamps corresponding to the training data
                    tX_train = deque(maxlen=training_size)
                    tX_train.extend(tX_init)
                    ty_train = deque(maxlen=training_size)
                    ty_train.extend(ty_init)
                    ty_train_ = deque(maxlen=training_size)
                    ty_train_.extend(ty_init_)
                    # initializing the acquisition vector
                    acquisitions = deque(maxlen=training_size)
                    acquisitions.extend(np.full(len(y_train), True))
                    # train the model with the initially available data
                    if hasattr(clf, 'partial_fit') and callable(getattr(clf, 'partial_fit')):
                        clf.partial_fit(X_train, y_train)
                    else:
                        clf.fit(X_train, y_train)
                    yhat = clf.predict(X_train)
                    # initialize the list that stores the result of the classifier's prediction
                    correct_classifications = [y_ == y for y_, y in zip(yhat, y_train)]
                    # init_acc = np.sum(correct_classifications) / len(yhat)
                    # print('Init class accuracy: {}'.format(init_acc))
                    # initialize the number of acquired labels
                    count = 0
                    # iterate over the whole data stream
                    for t, (x_t, y_t, tX_t, ty_t, ty_t_) in enumerate(zip(X_stream, y_stream, tX_stream, ty_stream,
                                                                          ty_stream_)):
                        # infer the currently available labels
                        # missing_label is used to denote unlabeled instances
                        X_cand = x_t.reshape([1, -1])
                        y_cand = y_t
                        tX_cand = tX_t
                        ty_cand = ty_t
                        ty_cand_ = ty_t_
                        # manage delay in label. Do not forget. Do not add unavailable labels due delay.
                        y_train_current = np.array(
                            [y if ty < tX_cand and a else missing_label for ty, y, a in
                             zip(ty_train, y_train, acquisitions)])

                        # # Remove nan for training, but do not reduce the training size
                        # no_nan = ~np.isnan(y_train_current)
                        # y_train_nonan.extend(y_train_current[no_nan])
                        # X_train_nonan.extend(np.array(X_train)[no_nan])

                        # test than train
                        # evaluate the prediction of the classifier
                        correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
                        # train the classifier
                        sample_weight = None
                        if np.sum(~np.isnan(y_train_current)) > 0:
                            # Propagate label
                            if 'Propagate' in delay_strategy_name:
                                y_train_ = propagate(np.array(X_train), np.array(y_train_current), k=5, ty=np.array(ty_train_))
                            else:
                                y_train_ = np.array(y_train_current)

                            # new_y_train = propagate_label(np.array(X_train), np.array(y_train_current), np.array(acquisitions))
                            # sample_weight = np.exp(-(np.arange(len(y_train_current), 0, -1) * 0.005) ** 2)
                            if hasattr(clf, 'partial_fit') and callable(getattr(clf, 'partial_fit')):
                                clf.partial_fit(np.array(X_train), y_train_,
                                                sample_weight=sample_weight)
                            else:
                                clf.fit(np.array(X_train), y_train_, sample_weight=sample_weight)

                        # check whether to sample the instance or not
                        sampled_indices, utilities = call_func(delay_wrapper.query, X_cand=X_cand, clf=clf,
                                                               X=np.array(X_train),
                                                               y=np.array(y_train_current),
                                                               tX=np.array(tX_train), ty=np.array(ty_train_),
                                                               tX_cand=[tX_cand],
                                                               ty_cand=[ty_cand_], return_utilities=True,
                                                               acquisitions=acquisitions)
                        # create budget_manager_param_dict for BIQF used by PALS
                        budget_manager_param_dict = {"utilities": utilities}
                        delay_wrapper.update(X_cand, sampled_indices, budget_manager_param_dict)
                        # set the entry within the acquisition vector according to the query strategy's decision
                        acquisitions.append((len(sampled_indices) > 0))
                        if len(sampled_indices):
                            count += 1

                        # add the current instance to the training data
                        tX_train.append(tX_cand)
                        ty_train.append(ty_cand)
                        ty_train_.append(ty_cand_)
                        X_train.append(x_t)
                        y_train.append(y_cand)
                    # calculate and show the average accuracy
                    avg_acc = np.sum(correct_classifications) / len(correct_classifications)
                    if args.verbose > 0:
                        print("Delay Wrapper: ", delay_strategy_name, ", Avg Accuracy: ", avg_acc,
                              ", Number of acquired instances: ", count)
                    # smoothing the accuracy for plotting
                    smoothing_window_length = 100
                    smoothed_curve = np.convolve(correct_classifications, np.ones(smoothing_window_length),
                                                 mode='valid') / smoothing_window_length  # len: max(M,N) - min(M,N) + 1
                    ax.plot(smoothed_curve, label=f'{delay_strategy_name}:{avg_acc:.3f}')
                    for d in drifts:
                        ax.axvline(x=d - smoothing_window_length, color='red', linestyle='-')
                    ax.set_title(f'{query_strategy_name}: B:{budget}, type:{args.latency_type}, '
                                 f'true:{verification_latency}, hat:{latency_hat}')

                    # append dataframe
                    df = df.append({'seed': args.init_seed,
                                    'verification_latency': verification_latency,
                                    'latency_tyep': args.latency_type,
                                    'delta_latency': delta_latency,
                                    'latency_hat': latency_hat,
                                    'query_strategy': query_strategy_name,
                                    'delay_strategy': delay_strategy_name,
                                    'avg_acc': avg_acc,
                                    'n_queried': count,
                                    'knn': args.k_neighbours,
                                    'weighted': args.weighted
                                    }, ignore_index=True)

                    iter_end_time = time.time()
                    iter_seconds = iter_end_time - iter_start_time
                    total_seconds = iter_end_time - start_time
                    print('Iteration {}/{} time: {} - ETA: {}'.format(n, tot_iter, time.strftime("%Mm:%Ss", time.gmtime(
                        iter_seconds)), time.strftime('%Hh:%Mm:%Ss',
                                                      time.gmtime((total_seconds / (n + 1)) * (tot_iter - (n + 1))))))
                    n += 1

                    if query_strategy_name == 'RandomSampler':
                        break
                ax.legend()
                saver.save_fig(fig, name=f'{query_strategy_name}')
                plt.close('all')
                if args.verbose > 0:
                    print()

    df['model'] = args.model
    df['dataset'] = args.dataset
    df.to_csv(os.path.join(saver.path, 'results_df.csv'), sep=',', index=False)
    return df
