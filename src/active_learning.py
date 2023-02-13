import argparse
import json
import logging
import sys
import warnings
from collections import deque

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from skactiveml.classifier import PWC
from skactiveml.stream import FixedUncertainty, VariableUncertainty, Split, PALS
from skactiveml.stream import RandomSampler, PeriodicSampler
from skactiveml.utils import call_func
from skactiveml.stream.budget_manager import FixedThresholdBudget, FixedUncertaintyBudget, BIQF, \
    VariableUncertaintyBudget, SplitBudget, EstimatedBudget
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.induce_concept_drift import induce_drift, corrupt_drift
from src.utils.log_utils import StreamToLogger
from src.utils.new_dataload import OUTPATH
from src.utils.training_helper_v3 import *

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns
sns.set_style("whitegrid")


######################################################################################################
class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


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


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    parser = argparse.ArgumentParser(description='Active Learning Example')

    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--informative', type=int, default=10)
    parser.add_argument('--redundant', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='blobs')
    parser.add_argument('--drift_features', type=str, default='top',
                        help='Which features to corrupt. Choices: top - bottom - list with features')
    parser.add_argument('--drift_type', type=str, default='gradual', help='flip or gradual')
    parser.add_argument('--drift_p', type=float, default=0.50, help='Percentage of features to corrupt')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--init_seed', type=int, default=0, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')

    # Add parameters for each particular network
    args = parser.parse_args()
    return args


######################################################################################################
if __name__ == '__main__':
    args = parse_args()
    print(args)
    print()

    ######################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = args.init_seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)

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
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0], network=os.path.join(args.dataset,
                                                                                                args.drift_type))

    # Save json of args/parameters. This is handy for TL
    with open(os.path.join(saver.path, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

        ######################################################################################################
        # Logging setting
        print('run logfile at: ', os.path.join(saver.path, 'logfile.log'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            filename=os.path.join(saver.path, 'logfile.log'),
            filemode='a'
        )

        # Redirect stdout
        stdout_logger = logging.getLogger('STDOUT')
        slout = StreamToLogger(stdout_logger, logging.INFO)
        sys.stdout = slout

        # Redirect stderr
        stderr_logger = logging.getLogger('STDERR')
        slerr = StreamToLogger(stderr_logger, logging.ERROR)
        sys.stderr = slerr

        # Suppress output
        if args.disable_print:
            slout.terminal = open(os.devnull, 'w')
            slerr.terminal = open(os.devnull, 'w')

        ######################################################################################################
        # Loading data
        n_features = args.features
        classes = args.classes
        cluster_std = 3
        if args.dataset == 'blobs':
            X, y, centers = make_blobs(n_samples=2000, n_features=n_features, centers=classes, cluster_std=cluster_std,
                                       return_centers=True)
            x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=n_features, centers=centers + 25,
                                                       cluster_std=1.0,
                                                       return_centers=True)
            Y_test_ = Y_test_[:, np.newaxis]
            x_test, Y_test = make_blobs(n_samples=1000, n_features=n_features, centers=centers, cluster_std=cluster_std)
            Y_test = Y_test[:, np.newaxis]
            if args.drift_type.lower() == 'gradual':
                for i in [x * 0.1 for x in range(1, 10, 1)]:
                    x_, y_ = make_blobs(n_samples=100, n_features=n_features,
                                        centers=linear_interp(centers_end, centers, i),
                                        cluster_std=1.0)
                    y_ = y_[:, np.newaxis]
                    x_test = np.vstack([x_test, x_])
                    Y_test = np.vstack([Y_test, y_])
            x_test = np.vstack([x_test, x_test_])
            Y_test = np.vstack([Y_test, Y_test_])
            Y_test = Y_test.squeeze(1)
            t_start = 500
            t_end = t_start

            x_train, x_valid, Y_train, Y_valid = train_test_split(X, y, shuffle=True, test_size=0.2)

            ## Data Scaling
            normalizer = StandardScaler()
            x_train = normalizer.fit_transform(x_train)
            x_valid = normalizer.transform(x_valid)
            x_test = normalizer.transform(x_test)

        elif args.dataset.lower() == 'rbf':
            X, y = make_classification(n_samples=5000, n_features=n_features, n_informative=args.informative,
                                       n_redundant=args.redundant, n_repeated=0, n_classes=classes,
                                       n_clusters_per_class=2, weights=None, flip_y=0.0, class_sep=10., hypercube=True,
                                       shift=0.0, scale=1.0, shuffle=True, random_state=args.init_seed)

            x_train, x_test, Y_train, Y_test = train_test_split(X, y, shuffle=False, train_size=0.6)
            x_train, x_valid, Y_train, Y_valid = train_test_split(x_train, Y_train, shuffle=False, test_size=0.2)

            ## Data Scaling
            normalizer = eval(args.preprocessing)()
            x_train = normalizer.fit_transform(x_train)
            x_valid = normalizer.transform(x_valid)
            x_test = normalizer.transform(x_test)

            train_samples = len(x_train)
            valid_samples = len(x_valid)
            test_samples = len(x_test)

            if args.drift_type == 'flip':
                t_start = int(0.60 * test_samples)
                t_end = t_start
                x_test, permute_dict = induce_drift(x_test, y=Y_test, t_start=t_start, t_end=None, p=args.drift_p,
                                                    features=args.drift_features, copy=False)
            elif args.drift_type == 'gradual':
                t_start = int(0.60 * test_samples)
                t_end = t_start + int(0.20 * test_samples)
                # t_end = t_start
                x_test, permute_dict = corrupt_drift(x_test, y=Y_test, t_start=t_start, t_end=t_end, p=args.drift_p,
                                                     features=args.drift_features, loc=1., std=1.0, copy=False)

        # Remove one class
        # x_train = x_train[Y_train!=3]
        # Y_train = Y_train[Y_train!=3]
        # x_valid = x_valid[Y_valid!=3]
        # Y_valid = Y_valid[Y_valid!=3]

        # Encode labels
        unique_train = np.unique(Y_train)
        label_encoder = dict(zip(unique_train, range(len(unique_train))))
        unique_test = np.setdiff1d(np.unique(y), unique_train)
        label_encoder.update(dict(zip(unique_test, range(len(unique_train), len(np.unique(y))))))

        Y_train = np.array([label_encoder.get(e, e) for e in Y_train])
        Y_valid = np.array([label_encoder.get(e, e) for e in Y_valid])
        Y_test = np.array([label_encoder.get(e, e) for e in Y_test])

        classes = len(np.unique(Y_train))

        ###########################
        # Induce Concept Drift
        train_samples = len(x_train)
        valid_samples = len(x_valid)
        test_samples = len(x_test)

        print('Num Classes: ', classes)
        print('Train:', x_train.shape, Y_train.shape, [(Y_train == i).sum() for i in np.unique(Y_train)])
        print('Validation:', x_valid.shape, Y_valid.shape,
              [(Y_valid == i).sum() for i in np.unique(Y_valid)])
        print('Test:', x_test.shape, Y_test.shape,
              [(Y_test == i).sum() for i in np.unique(Y_test)])

        ######################################################################################################
        # number of instances that are provided to the classifier
        init_train_length = 100
        # the length of the data stream
        stream_length = 4000
        # the size of the sliding window that limits the training data
        training_size = 500
        # query budget
        budget = 0.1
        # verification latency
        latency = 0

        # random state that is used to generate random seeds
        np.random.seed(0)
        random_state = np.random.RandomState(0)

        # X, center = make_blobs(n_samples=init_train_length + stream_length, centers=30,
        #                        random_state=get_randomseed(random_state), shuffle=True)
        # y = center % 2

        # Drifting Dataset
        X, y, centers = make_blobs(n_samples=1000 + init_train_length, n_features=2, centers=[(-3, 3), (3, 3)],
                                   cluster_std=1.,
                                   return_centers=True)
        x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=2, centers=centers + 25,
                                                   cluster_std=1.0, return_centers=True)
        X = np.vstack([X, x_test_])
        y = np.hstack([y, Y_test_])
        x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=2, centers=centers - 25,
                                                   cluster_std=1.0, return_centers=True)
        # Gradual drift
        # for i in [x * 0.1 for x in range(1, 10, 1)]:
        #    x_, y_ = make_blobs(n_samples=300, n_features=2,
        #                        centers=linear_interp(centers_end, centers, i),
        #                        cluster_std=1.0)
        #   X = np.vstack([X, x_])
        #   y = np.hstack([y, y_])
        # X = np.vstack([X, x_test_])
        # y = np.hstack([y, Y_test_])
        stream_length = len(X)

        X_init = X[:init_train_length, :]
        y_init = y[:init_train_length]
        X_stream = X[init_train_length:, :]
        y_stream = y[init_train_length:]

        ######################################################################################################
        from sklearn.svm import SVC
        from skactiveml.classifier import SklearnClassifier

        clf_factory = lambda: PWC(classes=[0, 1], random_state=get_randomseed(random_state))
        # clf_factory = lambda: SklearnClassifier(SVC(probability=True), classes=[0, 1], random_state=get_randomseed(random_state))

        query_strategies = {
            'RandomSampler': RandomSampler(random_state=get_randomseed(random_state),
                                           budget_manager=FixedThresholdBudget(budget=budget)),
            #'PeriodicSampler': PeriodicSampler(random_state=get_randomseed(random_state),
            #                                   budget_manager=FixedThresholdBudget(budget=budget)),
            #'FixedUncertainty': FixedUncertainty(random_state=get_randomseed(random_state),
            #                                     budget_manager=FixedUncertaintyBudget(budget=budget)),
            #'VariableUncertainty': VariableUncertainty(random_state=get_randomseed(random_state),
            #                                           budget_manager=VariableUncertaintyBudget(budget=budget)),
            'Split': Split(random_state=get_randomseed(random_state), budget_manager=SplitBudget(budget=budget)),
            'PALS': PALS(random_state=get_randomseed(random_state), budget_manager=BIQF(budget=budget))
        }

        acc_strategies = dict.fromkeys(query_strategies.keys())
        t_strategies = dict.fromkeys(query_strategies.keys())
        train_flag = False
        for query_strategy_name, query_strategy in query_strategies.items():
            latency_ = latency
            clf = clf_factory()
            # initializing the training data
            X_train = deque(maxlen=training_size)
            X_train.extend(X_init)
            y_train = deque(maxlen=training_size)
            y_train.extend(y_init)
            # train the model with the initially available data
            clf.fit(X_train, y_train)
            # initialize the list that stores the result of the classifier's prediction
            correct_classifications = []
            timestamps = []
            count = 0
            for t, (x_t, y_t) in enumerate(zip(X_stream, y_stream)):
                # create stream samples
                X_cand = x_t.reshape([1, -1])
                y_cand = y_t
                # train the classifier
                #if train_flag:
                clf.fit(X_train, y_train)
                correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
                # check whether to sample the instance or not
                # call_func is used since a classifier is not needed for RandomSampler and PeriodicSampler
                sampled_indices, utilities = call_func(query_strategy.query, X_cand=X_cand, clf=clf,
                                                       return_utilities=True)
                # create budget_manager_param_dict for BIQF used by PALS
                budget_manager_param_dict = {"utilities": utilities}
                # update the query strategy and budget_manager to calculate the right budget
                query_strategy.update(X_cand, sampled_indices, budget_manager_param_dict)
                # count the number of queries
                count += len(sampled_indices)
                # add queried timestamp
                if len(sampled_indices) > 0:
                    timestamps.append(t)
                # add X_cand to X_train
                X_train.append(x_t)
                # add label or missing_label to y_train
                y_train.append(y_cand if len(sampled_indices) > 0 else clf.missing_label)
                #train_flag = True if len(sampled_indices) > 0 else False
            # calculate and show the average accuracy
            print("Query Strategy: ", query_strategy_name, ", Avg Accuracy: ", np.mean(correct_classifications),
                  ", Acquisation count:", count)
            acc_strategies[query_strategy_name] = correct_classifications
            t_strategies[query_strategy_name] = timestamps

            #plt.plot(gaussian_filter1d(np.array(correct_classifications, dtype=float), 10), label=query_strategy_name)
            smoothing_window_length = 100
            smoothed_curve = np.convolve(correct_classifications, np.ones(smoothing_window_length),
                                         mode='valid') / smoothing_window_length  # len: max(M,N) - min(M,N) + 1
            plt.plot(smoothed_curve, label=query_strategy_name)

        plt.legend()
    plt.show(block=True)
