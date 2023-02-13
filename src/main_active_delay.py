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

from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier

from src.concept_drift.skflow_detectors import HDDDM

from src.utils.saver import Saver
from src.utils.new_dataload import OUTPATH, DATAPATH
from src.KDE_uncert.drifting_dataset import create_dataset

from src.KDE_uncert.single_exp_AL import single_experiment

warnings.filterwarnings("ignore")
columns = shutil.get_terminal_size().columns


######################################################################################################
class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def remove_duplicates(sequence):
    unique = []
    [unique.append(item) for item in sequence if item not in unique]
    return unique


def check_ziplen(l, n):
    if len(l) % n != 0:
        l += [l[-1]]
        return check_ziplen(l, n)
    else:
        return l


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def linear_interp(a, b, alpha):
    return a * alpha + (1 - alpha) * b


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run(x, args, path, result_path):
    print('Process PID:', os.getpid())
    args.init_seed = x

    df_run = single_experiment(args, path)
    if os.path.exists(result_path):
        df_run.to_csv(result_path, mode='a', sep=',', header=False, index=False)
    else:
        df_run.to_csv(result_path, mode='a', sep=',', header=True, index=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='blobs')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_concept_drifts', type=int, default=2)

    parser.add_argument('--model', type=str, default='PWC', choices=('PWC', 'SVC', 'KNN', 'HT', 'NB'))
    parser.add_argument('--query_strategy', type=str, nargs='+',
                        default=['RandomSampler', "Split", "PALS"],
                        choices=(
                            'RandomSampler', 'PeriodicSampler', 'FixedUncertainty', 'VariableUncertainty', 'Split',
                            'PALS'))
    parser.add_argument('--delay_strategy', type=str, nargs='+',
                        default=['None', 'Propagate'],
                        choices=('None', 'Forgetting', 'Propagete', 'Bagging'))

    parser.add_argument('--k_neighbours', type=int, default=5)
    parser.add_argument('--weighted', action='store_true', default=True)

    parser.add_argument('--init_train_length', type=int, default=10)
    parser.add_argument('--stream_length', type=int, default=2000, help='This parameter is also sample per concept')
    parser.add_argument('--training_size', type=int, default=500)
    # parser.add_argument('--min_train_size', type=int, default=100)
    parser.add_argument('--verification_latency', type=int, nargs='+', default=[300],
                        help='Multiple arguments for multiple analysis. [0, 50, 100, 200, 300]')
    parser.add_argument('--delta_latency', type=int, nargs='+', default=[0])
    parser.add_argument('--latency_type', type=str, default='fixed', choices=('fixed', 'batch', 'random'))
    parser.add_argument('--budget', type=float, default=0.1)

    # Delay wrapper parameters
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--delay_prior', type=float, default=0.001)

    parser.add_argument('--init_seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=3, help='Number or runs')

    parser.add_argument('--process', type=int, default=5, help='Number of parallel process. Single GPU.')

    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()
    return args


######################################################################################################
def main():
    args = parse_args()
    print(args)
    print()

    ######################################################################################################
    SEED = args.init_seed
    np.random.seed(SEED)

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        plt.switch_backend(backend)
    print()

    ######################################################################################################
    # LOG STUFF
    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  network=os.path.join(args.dataset, args.model))
    saver.make_log(**vars(args))
    csv_path = os.path.join(saver.path, '{}_{}_results.csv'.format(args.dataset, args.model))

    ######################################################################################################
    seeds_list = list(np.random.choice(1000, args.n_runs, replace=False))
    seeds_list = check_ziplen(seeds_list, args.process)
    total_run = args.n_runs

    iterator = zip(*[seeds_list[j::args.process] for j in range(args.process)])
    total_iter = len(list(iterator))

    # mp.set_start_method('spawn', force=True)
    start_datetime = datetime.now()
    print('{} - Start Experiments. {} parallel process.  \r\n'.format(
        start_datetime.strftime("%d/%m/%Y %H:%M:%S"), args.process))
    for i, (x) in enumerate(zip(*[seeds_list[j::args.process] for j in range(args.process)])):
        start_time = time.time()
        x = remove_duplicates(x)
        n_process = len(x)
        idxs = [i * args.process + j for j in range(1, n_process + 1)]
        print('/' * shutil.get_terminal_size().columns)
        print(
            'ITERATION: {}/{}'.format(idxs, total_run).center(columns))
        print('/' * shutil.get_terminal_size().columns)
        print(f'Seed: {x}')

        process = []
        for j in range(n_process):
            process.append(mp.Process(target=run, args=(x[j], copy.deepcopy(args), saver.path, csv_path)))

        for p in process:
            p.start()

        for p in process:
            p.join()

        end_time = time.time()
        iter_seconds = end_time - start_time
        total_seconds = end_time - start_datetime.timestamp()
        print('Iteration time: {} - ETA: {}'.format(time.strftime("%Mm:%Ss", time.gmtime(iter_seconds)),
                                                    time.strftime('%Hh:%Mm:%Ss',
                                                                  time.gmtime(
                                                                      total_seconds * (total_iter / (i + 1) - 1)))))
        print()

    print('*' * shutil.get_terminal_size().columns)
    print('DONE!')
    end_datetime = datetime.now()
    total_seconds = (end_datetime - start_datetime).total_seconds()
    print('{} - Experiment took: {}'.format(end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                                            time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_seconds))))
    print(f'results dataframe saved in: {csv_path}')

    # Postprocess
    df = pd.read_csv(csv_path)
    import seaborn as sns
    g = sns.catplot(data=df, x='query_strategy', y='avg_acc', hue='delay_strategy', col='verification_latency',
                    kind='box', row='delta_latency')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Dataset: {} - Model: {}'.format(args.dataset, args.model))
    plt.savefig(os.path.join(saver.path, 'results.png'))

    df['rank'] = df.groupby(['verification_latency', 'delta_latency'])['avg_acc'].rank(method='average', ascending=True,
                                                                                       pct=True)
    g = sns.relplot(data=df, x='verification_latency', y='rank', col='delta_latency', hue='query_strategy',
                style='delay_strategy', kind='line', err_style="bars", markers=True, err_kws={'capsize':3})
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Dataset: {} - Model: {}'.format(args.dataset, args.model))
    plt.savefig(os.path.join(saver.path, 'results_rank.png'))

if __name__ == '__main__':
    main()
