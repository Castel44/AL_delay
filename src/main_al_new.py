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
from src.utils.new_dataload import OUTPATH, CLUSTER_RESULTS
from src.KDE_uncert.drifting_dataset import create_dataset

from src.single_exp_AL_new import single_experiment

warnings.filterwarnings("ignore")
columns = shutil.get_terminal_size().columns


######################################################################################################
class SaverSlave(Saver):
    def __init__(self, path, i):
        super(Saver)

        self.path = path
        self.i = i
        self.makedir_()
        # self.make_log()

    def makedir_(self):
        i = self.i
        path = os.path.join(self.path, f'exp_{i}')
        os.makedirs(path, exist_ok=False)
        self.path = path
        return self


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


def run(x, args, path):
    pid = os.getpid()
    print('Process PID:', pid)
    args.init_seed = x

    df_run = single_experiment(args, path)
    df_run.to_csv(os.path.join(path, f'{pid}.csv'), index=False)


def merge_csv(path):
    df = pd.DataFrame()
    for root, subdirectories, files, in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                df_tmp = pd.read_csv(os.path.join(root, file))
                df = pd.concat([df, df_tmp.copy()], ignore_index=True)
                os.remove(os.path.join(root, file))
    df.to_csv(os.path.join(path, 'results.csv'), index=False)
    print(f"results dataframe saved in: {os.path.join(path, 'results.csv')}")
    return os.path.join(path, 'results.csv')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SYNTH_2_2',
                        help="<SYNTH or BLOBS>_<features>_<classes> or one of available data")
    parser.add_argument('--n_concept_drifts', type=int, default=1)
    parser.add_argument('--induced_drift_type', type=str, default='flip', help='flip or corrupt')

    parser.add_argument('--model', type=str, default='PWC', choices=('PWC', 'SVC', 'KNN', 'HT', 'NB'))

    parser.add_argument('--algo', type=str,
                        default=["PR+pal"], nargs='+',
                        choices=(
                            "rand", "var_uncer", "split", "pal",
                            "FO+var_uncer", "FO+split", "FO+pal",
                            "BI+var_uncer", "BI+split", "BI+pal",
                            "FI+var_uncer", "FI+split", "FI+pal",
                            "FO+BI+var_uncer", "FO+BI+split", "FO+BI+pal",
                            "FO+FI+var_uncer", "FO+FI+split", "FO+FI+pal",
                            "PR+var_uncer", "PR+split", "PR+pal"))

    parser.add_argument('--drift_detector', type=str, default='None',
                        choices=('None', 'DDM', 'EDDM', 'PH', 'ADWIN', 'HDDDM'))

    parser.add_argument('--dynamic_window', action='store_true', default=False)
    parser.add_argument('--dynamic_budget', action='store_true', default=True)
    parser.add_argument('--modified_training', action='store_true', default=False)

    parser.add_argument('--bplus', type=int, default=4)
    parser.add_argument('--bmin', type=int, default=2)

    parser.add_argument('--k_neighbours', type=int, default=3)
    parser.add_argument('--l', type=float, default=0.01)

    parser.add_argument('--init_train_length', type=int, default=100)
    parser.add_argument('--stream_length', type=int, default=1000, help='This parameter is also sample per concept')
    parser.add_argument('--training_size', type=int, default=200)
    parser.add_argument('--min_training_size', type=int, default=100, help='Only if drift detector is not None')
    parser.add_argument('--verification_latency', type=int, nargs='+', default=[10],
                        help='Multiple arguments for multiple analysis. [0, 50, 100, 200, 300]')
    parser.add_argument('--delta_latency', type=int, nargs='+', default=[0])
    parser.add_argument('--latency_type', type=str, default='normal', choices=('fixed', 'batch', 'uniform', 'normal'))
    parser.add_argument('--budget', type=float, nargs='+', default=[0.1])

    # Delay wrapper parameters
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--delay_prior', type=float, default=0.001)
    parser.add_argument('--prior_exp', type=float, default=0)

    parser.add_argument('--init_seed', type=int, default=420)
    parser.add_argument('--n_runs', type=int, default=10, help='Number or runs')
    parser.add_argument('--process', type=int, default=5, help='Number of parallel process. Single GPU.')
    parser.add_argument('--dt_exp', type=str, default="None", help='Datetime of start experiment')
    parser.add_argument('--iter_idx', type=int, default=0, help='Iteration index for folder name. Only used in cluster')

    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=True, help='Matplotlib backend')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--cluster', action='store_true', default=False, help='do not plot')

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

    if args.headless or args.cluster:
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
    if args.cluster:
        saver = SaverSlave(
            os.path.join(CLUSTER_RESULTS, args.dt_exp, os.path.basename(__file__).split(sep='.py')[0], args.dataset,
                         args.model), args.iter_idx)
    else:
        saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                      network=os.path.join(args.dataset, args.model))
    saver.make_log(**vars(args))
    print(f"Experiment path: {saver.path}")

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
            process.append(mp.Process(target=run, args=(x[j], copy.deepcopy(args), saver.path)))

        for p in process:
            p.start()
            # time.sleep(0.1)

        for p in process:
            p.join()
            # time.sleep(0.1)

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
    saver.append_str(['#' * 80, "Experiment time: {}".format(time.strftime("%Hh:%Mm:%Ss", time.gmtime(total_seconds)))])

    # Postprocess
    if not args.cluster:
        csv_path = merge_csv(saver.path)
        df = pd.read_csv(csv_path)
        import seaborn as sns
        g = sns.catplot(data=df, x='verification_latency', y='avg_acc', hue='algo', kind='box', row='budget',
                        col='delta_latency')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Dataset: {} - Model: {} - Latency: {}'.format(args.dataset, args.model, args.latency_type))
        plt.savefig(os.path.join(saver.path, 'results.png'))


if __name__ == '__main__':
    main()
