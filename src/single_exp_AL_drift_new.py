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
from skactiveml.stream.budget_manager import FixedThresholdBudget, FixedUncertaintyBudget, \
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

from src.stream_al.util.split_training_data import split_training_data, get_training_data, LABEL_NOT_SELECTED

from src.stream_al.classifier.pwc_wrapper import PWCWrapper

from src.stream_al.budget_manager.biqf import BIQF
from src.stream_al.budget_manager.fixed_budget import FixedBudget
from src.stream_al.budget_manager.uncertainty_budget import FixedUncertaintyBudget, VarUncertaintyBudget, SplitBudget

from src.stream_al.selection_strategies.baseline import RandomSelection, PeriodicSample
from src.stream_al.selection_strategies.uncertainty import Uncertainty
from src.stream_al.selection_strategies.delay_wrapper import FOWrapper, BIWrapper, FIWrapper
from src.stream_al.selection_strategies.pal import PAL_BIQF

from src.KDE_uncert.load_datasets import load_raw_dataset_by_name

from src.KDE_uncert.propagate_wrapper_new import PRWrapper

from src.stream_al.util.split_training_data import split_training_data, get_training_data, LABEL_NOT_SELECTED, \
    get_training_data_modified

from scipy.spatial.distance import cdist

from skmultiflow.drift_detection import PageHinkley, KSWIN, EDDM, DDM, HDDM_W, HDDM_A
from skmultiflow.drift_detection.adwin import ADWIN
from src.concept_drift.skflow_detectors import HDDDM


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


def get_active_learner_by_name(name, **kwargs):
    name_parts = name.split('+')
    first = name_parts[0]
    rest = '+'.join(name_parts[1:])

    base_selection_strategy = None
    base_budget_manager = None
    if len(rest):
        base_selection_strategy, base_budget_manager = get_active_learner_by_name(rest, **kwargs)

    al_selection_strategies = {}
    al_selection_strategies['rand'] = lambda: (
        RandomSelection(kwargs['rand']), BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))
    al_selection_strategies['periodic_sample'] = lambda: (
        PeriodicSample(kwargs['budget']), BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))

    al_selection_strategies['fixed_uncer'] = lambda: (Uncertainty(kwargs['clf_factory_function']),
                                                      FixedUncertaintyBudget(budget=kwargs['budget'], w=kwargs['w'],
                                                                             n_classes=kwargs['n_classes']))
    al_selection_strategies['var_uncer'] = lambda: (Uncertainty(kwargs['clf_factory_function']),
                                                    VarUncertaintyBudget(budget=kwargs['budget'], w=kwargs['w'],
                                                                         theta=kwargs['theta'], s=kwargs['s']))
    al_selection_strategies['split'] = lambda: (Uncertainty(kwargs['clf_factory_function']),
                                                SplitBudget(budget=kwargs['budget'], w=kwargs['w'], rand=kwargs['rand'],
                                                            v=kwargs['v'], theta=kwargs['theta'], s=kwargs['s']))

    al_selection_strategies['pal'] = lambda: (
        PAL_BIQF(kwargs['pwc_factory_function'], kwargs['prior'], kwargs['m_max']),
        BIQF(budget=kwargs['budget'], w=kwargs['w'], w_tol=kwargs['w_tol']))

    al_selection_strategies['FO'] = lambda: (
        FOWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy,
                  delay_future_buffer=kwargs['delay_future_buffer']), base_budget_manager)
    al_selection_strategies['BI'] = lambda: (
        BIWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy, K=kwargs['K'],
                  delay_prior=kwargs['delay_prior'], pwc_factory_function=kwargs['pwc_factory_function']),
        base_budget_manager)
    al_selection_strategies['FI'] = lambda: (
        FIWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy,
                  delay_prior=kwargs['delay_prior'], pwc_factory_function=kwargs['pwc_factory_function']),
        base_budget_manager)

    al_selection_strategies['PR'] = lambda: (
        PRWrapper(random_state=kwargs['rand'], base_selection_strategy=base_selection_strategy, k=kwargs['k'],
                  l=kwargs['l']),
        base_budget_manager)

    selection_strategy, budget_manager = al_selection_strategies[first]()
    return selection_strategy, budget_manager


def get_clf_by_name(name, n_classes, random_state, training_size):
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
    return clf_factory[name]


def get_drift_detector(name, **kwargs):
    drift_factory = {
        'None': lambda: None,
        'DDM': lambda: DDM(min_num_instances=50, warning_level=2., out_control_level=3.),
        'ADWIN': lambda: ADWIN(delta=1.5),  # higher delta the more sensitive
        'EDDM': lambda: EDDM(),
        'PH': lambda: PageHinkley(min_instances=50, delta=0.005, threshold=20, alpha=1 - 0.0001),
        'HDDDM': lambda: HDDDM(window_size=500, min_samples=250, warning_gamma=1.,
                               change_gamma=1.5, dimension=kwargs['dimension'])
    }
    return drift_factory[name]


def get_tx_ty(type, verification_latency, delta_latency, stream_length):
    # TODO: fix
    if verification_latency == 0:
        latency = 0
    else:
        if type == 'fixed':
            latency = verification_latency

        elif type == 'batch':
            latency = batch_latency(stream_length, verification_latency)

        elif type == 'uniform':
            latency = np.random.uniform(0, verification_latency, stream_length).astype(int)

        elif type == 'normal':
            latency = np.random.normal(verification_latency, 250, stream_length).astype(int).clip(0)

        else:
            raise ValueError
    print(f'latency type:{type}, {verification_latency=}, {delta_latency=}')
    tX = np.arange(stream_length)
    ty = tX + latency  # true latency
    ty_ = tX + delta_latency  # predicted latency
    return tX, ty, ty_


def new_labeled_predictions(tx_n, XT_dict, YT_dict, TY_dict):
    TX = np.array(list(XT_dict.keys()))
    YT = np.array([YT_dict[tx] for tx in TX])
    TY = np.array([TY_dict[tx] for tx in TX])

    idx = np.array(np.where(TY == tx_n)[0])
    idx_selected = idx[YT[idx] != -1]
    return TX[idx_selected]


def single_experiment(args, path):
    path = os.path.join(path, 'seed_{}'.format(args.init_seed))
    np.random.seed(args.init_seed)
    random_state = np.random.RandomState(args.init_seed)

    # Suppress output
    if args.disable_print:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    # Results dataframe
    df = pd.DataFrame()

    ######################################################################################################
    X_data, y_data, drifts = create_dataset(args.dataset, args.n_concept_drifts, p=0.7,
                                            type=args.induced_drift_type)

    stream_length = len(X_data)
    init_train_length = args.init_train_length
    n_classes = len(np.unique(y_data))
    n_features = X_data.shape[1]
    training_size = args.training_size
    w_train = training_size
    # Not used
    delay_future_buffer = 0

    ######################################################################################################
    # Define methods and strategies
    clf_model = get_clf_by_name(args.model, n_classes, random_state, training_size)

    ######################################################################################################
    verification_latency_pool = args.verification_latency
    delta_latency_pool = args.delta_latency
    algo_pool = args.algo
    budget_pool = args.budget
    tot_iter = len(verification_latency_pool) * len(delta_latency_pool) * len(algo_pool) * len(budget_pool)

    n = 0
    smoothing_window_length = 100
    start_time = time.time()
    for budget in budget_pool:
        for verification_latency in verification_latency_pool:
            for delta_latency in delta_latency_pool:
                # sample arrival
                tX_orig, ty_orig, ty_pred_orig = get_tx_ty(args.latency_type, verification_latency, delta_latency,
                                                           stream_length)
                saver = SaverSlave(path)

                for algo in algo_pool:
                    iter_start_time = time.time()

                    clf = clf_model()
                    detectors = {x: get_drift_detector(x, dimension=n_features)() for x in args.drift_detector}
                    pwc_wrap = lambda: PWCWrapper(classes=np.arange(n_classes), random_state=random_state,
                                                  N=budget * (w_train - verification_latency))
                    active_learner, budget_manager = get_active_learner_by_name(
                        name=algo,
                        rand=random_state,
                        budget=budget,
                        n_classes=n_classes,
                        n_features=n_features,
                        theta=1,
                        s=0.01,
                        v=0.1,
                        K=args.K,
                        k=args.k_neighbours,
                        l=args.l,
                        prior=1e-3,
                        m_max=3,
                        n_max=5,
                        delay_prior=10 ** args.prior_exp,
                        prior_c=1e-3,
                        prior_e=1e-3,
                        w=256,
                        w_tol=int(200 * budget),
                        delay_future_buffer=delay_future_buffer,
                        pwc_factory_function=pwc_wrap,
                        clf_factory_function=pwc_wrap
                    )

                    # train the model with the initially available data
                    X_init = X_data[:init_train_length, :]
                    y_init = y_data[:init_train_length]
                    X_stream = X_data[init_train_length:, :]
                    y_stream = y_data[init_train_length:]

                    XT_dict = {t: x.reshape([1, -1]) for t, x in enumerate(X_init)}
                    YT_dict = {t: x for t, x in enumerate(y_init)}
                    TY_dict = {t: x for t, x in enumerate(range(init_train_length))}
                    tX = tX_orig[init_train_length:]
                    ty = ty_orig[init_train_length:]
                    ty_pred = ty_pred_orig[init_train_length:]

                    if hasattr(clf, 'partial_fit') and callable(getattr(clf, 'partial_fit')):
                        clf.partial_fit(X_init, y_init)
                    else:
                        clf.fit(X_init, y_init)
                    yhat = clf.predict(X_init)
                    # initialize the list that stores the result of the classifier's prediction
                    correct_classifications = [y_ == y for y_, y in zip(yhat, y_init)]
                    known_acc = {x: y_ == y for x, y_, y in zip(range(len(yhat)), yhat, y_init)}

                    # # Initialize drift detectors with iniital samples
                    # for dd in detectors.keys():
                    #     if dd != 'None':
                    #         if dd == 'HDDDM':
                    #             for x in X_init:
                    #                 detectors[dd].add_element(x)
                    #         else:
                    #             err_sig = 1 - (np.asarray(correct_classifications).astype(int))
                    #             for e in err_sig:
                    #                 detectors[dd].add_element(e)

                    # initialize the number of acquired labels
                    count = 0
                    drift_idx = {x:[] for x in detectors.keys()}
                    budget_history = []

                    # iterate over the whole data stream
                    for t, (x, y, tx_n, ty_n, ty_n_pred) in enumerate(zip(X_stream, y_stream, tX, ty, ty_pred),
                                                                      start=init_train_length):
                        x = x.reshape([1, -1])
                        X_n, Lx_n, Ly_n, Lsw_n = get_training_data(tx_n, w_train, n_features, XT_dict, YT_dict, TY_dict)

                        # evaluate the prediction of the classifier
                        correct_classifications.append(clf.predict(x)[0] == y)
                        # train the classifier
                        sample_weight = None
                        if len(Lx_n) > 10:
                            if hasattr(clf, 'partial_fit') and callable(getattr(clf, 'partial_fit')):
                                clf.partial_fit(Lx_n, Ly_n, sample_weight=sample_weight)
                            else:
                                clf.fit(Lx_n, Ly_n, sample_weight=sample_weight)

                        al_score = active_learner.utility(
                            x,
                            copy.deepcopy(clf),
                            X_n=X_n,
                            Lx_n=Lx_n,
                            Ly_n=Ly_n,
                            Lsw_n=Lsw_n,
                            tx_n=tx_n,
                            ty_n=ty_n,
                            ty_n_pred=ty_n_pred,
                            w_train=w_train,
                            n_features=n_features,
                            XT_dict=XT_dict,
                            YT_dict=YT_dict,
                            TY_dict=TY_dict,
                            modified_training_data=False,
                            add_X=None,
                            add_Y=None,
                            add_SW=None,
                        )

                        budget_history.append(budget_manager.budget)
                        sampled = budget_manager.query(al_score)
                        active_learner.partial_fit(
                            x,
                            sampled,
                            clf=copy.deepcopy(clf)
                        )
                        if sampled[0]:
                            count += 1

                        # Update train window
                        XT_dict[tx_n] = x
                        TY_dict[tx_n] = ty_n
                        if sampled[0]:
                            YT_dict[tx_n] = y
                        else:
                            YT_dict[tx_n] = LABEL_NOT_SELECTED

                        # Detect drift
                        new_arrived_label = new_labeled_predictions(tx_n, XT_dict, YT_dict, TY_dict)
                        for dd in detectors.keys():
                            if dd != 'None':
                                if dd == 'HDDDM':
                                    detectors[dd].add_element(x)
                                else:
                                    for i_new in new_arrived_label:
                                        e = 1 - (correct_classifications[i_new]).astype(int)
                                        known_acc[tx_n] = correct_classifications[i_new]
                                        detectors[dd].add_element(e)
                                if detectors[dd].detected_change():
                                    if tx_n > drifts[0]:
                                        drift_idx[dd].append(tx_n)
                                        detectors[dd].reset()

                        # Remove outdated sample. Keep fixed training window size
                        if len(XT_dict) > w_train:
                            TX = np.array(list(XT_dict.keys()))
                            key_to_pop = TX[:-w_train]
                            for key in key_to_pop:
                                XT_dict.pop(key, None)
                                YT_dict.pop(key, None)
                                TY_dict.pop(key, None)

                    # calculate and show the average accuracy
                    avg_acc = np.sum(correct_classifications) / len(correct_classifications)
                    if args.verbose > 0:
                        print("Alg: ", algo, ", Avg Accuracy: ", avg_acc,
                              ", Number of acquired instances: ", count)

                    # smoothing the accuracy for plotting
                    smoothed_curve = np.convolve(correct_classifications, np.ones(smoothing_window_length),
                                                 mode='valid') / smoothing_window_length  # len: max(M,N) - min(M,N) + 1
                    # calculate recovery time
                    # suppose single drift
                    w = 600
                    recovery_time = []
                    for d in drifts:
                        real_drift_idx = d
                        avg_acc_pre = np.mean(correct_classifications[real_drift_idx-w:real_drift_idx])
                        std_acc_pre = np.std(correct_classifications[real_drift_idx-w:real_drift_idx])
                        recovery_time_ = np.where(smoothed_curve[real_drift_idx:] >= avg_acc_pre)[0]
                        if len(recovery_time_):
                            recovery_time.append(recovery_time_[0])
                        else:
                            recovery_time.append(1000)

                    fig, ax = plt.subplots()
                    ax.plot(smoothed_curve, label=f'{algo}:{avg_acc:.3f}')
                    for d,r in zip(drifts, recovery_time):
                        ax.axvline(x=d - smoothing_window_length, color='red', linestyle='-')
                        ax.axvline(x=r+d, color='green', linestyle='--', label=f'recovery_time: {r:.0f}')
                    for dd_name, d_idx in drift_idx.items():
                        idx_centered = [x - smoothing_window_length for x in d_idx]
                        ax.plot(idx_centered, smoothed_curve[idx_centered], '*', markersize=20, label=dd_name)
                    ax.set_title(f'Algo:{algo} - B:{budget}, type:{args.latency_type}, '
                                 f'true:{verification_latency}, delta:{delta_latency}')
                    ax.legend()
                    saver.save_fig(fig, name=f'accuracy')
                    plt.close('all')

                    # TODO: drift detection results

                    # append dataframe
                    df = df.append({'seed': args.init_seed,
                                    'verification_latency': verification_latency,
                                    'latency_type': args.latency_type,
                                    'delta_latency': delta_latency,
                                    'algo': algo,
                                    'avg_acc': avg_acc,
                                    'n_queried': count,
                                    'knn': args.k_neighbours,
                                    'l': args.l,
                                    'drift_idx': drift_idx,
                                    'budget': budget,
                                    'model': args.model,
                                    'dataset': args.dataset,
                                    'classes': n_classes,
                                    'features': n_features,
                                    'stream_length': stream_length,
                                    'training_size': training_size,
                                    'recovery_time': recovery_time[0],
                                    'real_drift': drifts[0]
                                    }, ignore_index=True)

                    iter_end_time = time.time()
                    iter_seconds = iter_end_time - iter_start_time
                    total_seconds = iter_end_time - start_time
                    print('Iteration {}/{} time: {} - ETA: {}'.format(n + 1, tot_iter,
                                                                      time.strftime("%Mm:%Ss",
                                                                                    time.gmtime(iter_seconds)),
                                                                      time.strftime('%Hh:%Mm:%Ss', time.gmtime(
                                                                          (total_seconds / (n + 1)) * (
                                                                                  tot_iter - (n + 1))))))
                    n += 1

                if args.verbose > 0:
                    print()

    return df
