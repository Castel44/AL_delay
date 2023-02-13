import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import deque

from scipy.ndimage import gaussian_filter1d
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

warnings.filterwarnings("ignore")
plt.switch_backend('Qt5Agg')


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def linear_interp(a, b, alpha):
    return a * alpha + (1 - alpha) * b


######################################################################################################
if __name__ == '__main__':
    np.random.seed(0)
    # random state that is used to generate random seeds
    random_state = np.random.RandomState(0)
    # number of instances that are provided to the classifier
    init_train_length = 100
    # the length of the data stream
    stream_length = 1000
    # the size of the sliding window that limits the training data
    training_size = 500
    # the verification latency occuring after querying a label
    verification_latency = 100
    budget = 0.1

    # Parameters for delay wrapper
    K = 2
    w_train = training_size
    delay_prior = 0.001

    # create the data stream
    # X, center = make_blobs(n_samples=init_train_length + stream_length, centers=30,
    #                       random_state=get_randomseed(random_state), shuffle=True)
    # y = center % 2

    # X, y = make_classification(n_samples=init_train_length + stream_length, n_features=2, n_informative=2,
    #                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0,
    #                           random_state=get_randomseed(random_state), shuffle=True)

    # Drifting Dataset
    X, y, centers = make_blobs(n_samples=1000 + init_train_length, n_features=2, centers=[(-3, 3), (3, 3)], cluster_std=1.,
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
    # create the time stamps
    # sample arrival
    tX = np.arange(stream_length)
    # label arrival
    ty = tX + verification_latency
    tX_init = tX[:init_train_length]
    ty_init = ty[:init_train_length]
    tX_stream = tX[init_train_length:]
    ty_stream = ty[init_train_length:]

    ######################################################################################################
    drift_factory = {
        'None': lambda qs: qs,
        'DDM': lambda: DDM(min_num_instances=30 + verification_latency, warning_level=2.0, out_control_level=3.0),
        'ADWIN': lambda: ADWIN(delta=2.),
        'EDDM': lambda: EDDM(),
        'PH': lambda: PageHinkley(min_instances=30 + verification_latency, delta=0.005, threshold=3, alpha=1 - 0.0001),
        'HDDDM': lambda: HDDDM(window_size=training_size, min_samples=training_size // 2, warning_gamma=1.,
                               change_gamma=2., dimension=X.shape[1])
    }

    ######################################################################################################
    # clf_factory = lambda: PWC(classes=[0, 1], random_state=get_randomseed(random_state))
    # clf_factory = lambda: SklearnClassifier(SVC(probability=True), classes=[0, 1], random_state=get_randomseed(random_state))
    clf_factory = lambda: SklearnClassifier(HoeffdingTreeClassifier(), classes=[0, 1], random_state=get_randomseed(random_state))
    # clf_factory = lambda : SklearnClassifier(KNNClassifier(n_neighbors=5, max_window_size=training_size), classes=[0, 1], random_state=get_randomseed(random_state))

    missing_label = clf_factory().missing_label
    query_strategies_factories = {
        'RandomSampler': lambda: RandomSampler(random_state=get_randomseed(random_state),
                                               budget_manager=FixedThresholdBudget(budget=budget)),
        # 'PeriodicSampler': lambda: PeriodicSampler(random_state=get_randomseed(random_state),
        #                                           budget_manager=FixedThresholdBudget(budget=budget)),
        # 'FixedUncertainty': lambda: FixedUncertainty(random_state=get_randomseed(random_state),
        #                                              budget_manager=FixedUncertaintyBudget(budget=budget)),
        #'VariableUncertainty': lambda: VariableUncertainty(random_state=get_randomseed(random_state),
        #                                                   budget_manager=VariableUncertaintyBudget(budget=budget)),
        'Split': lambda: Split(random_state=get_randomseed(random_state), budget_manager=SplitBudget(budget=budget)),
        'PALS': lambda: PALS(random_state=get_randomseed(random_state), budget_manager=BIQF(budget=budget))
    }

    for query_strategy_name, query_strategy_factory in query_strategies_factories.items():
        delay_wrappers_factories = {
            'None': lambda qs: qs,
            #'Forgetting': lambda qs: ForgettingWrapper(base_query_strategy=qs, w_train=w_train,
            #                                           random_state=get_randomseed(random_state)),
            # 'BaggingDelaySimulation': lambda qs: BaggingDelaySimulationWrapper(base_query_strategy=qs,
            #                                                                   random_state=get_randomseed(
            #                                                                       random_state), K=K,
            #                                                                   delay_prior=delay_prior),
            # 'FuzzyDelaySimulation': lambda qs: FuzzyDelaySimulationWrapper(base_query_strategy=qs,
            #                                                               random_state=get_randomseed(random_state),
            #                                                               delay_prior=delay_prior)
        }
        # delay_wrappers_factories["Forgetting + Bagging"] = lambda qs: delay_wrappers_factories["Forgetting"](
        #     BaggingDelaySimulationWrapper(base_query_strategy=qs,
        #                                   random_state=get_randomseed(
        #                                       random_state), K=K,
        #                                   delay_prior=delay_prior))
        # delay_wrappers_factories["Forgetting + Fuzzy"] = lambda qs: delay_wrappers_factories["Forgetting"](
        #    delay_wrappers_factories["FuzzyDelaySimulation"](qs))
        print("Query Strategy: ", query_strategy_name, )
        plt.figure()
        for delay_wrapper_name, delay_wrapper_factory in delay_wrappers_factories.items():
            u_drift = drift_factory['HDDDM']()
            s_drift = drift_factory['PH']()
            clf = clf_factory()
            # initializing the query strategy
            delay_wrapper = delay_wrapper_factory(query_strategy_factory())
            # initializing the training data
            X_train = deque(maxlen=training_size)
            X_train.extend(X_init)
            y_train = deque(maxlen=training_size)
            y_train.extend(y_init)
            # initialize the time stamps corresponding to the training data
            tX_train = deque(maxlen=training_size)
            tX_train.extend(tX_init)
            ty_train = deque(maxlen=training_size)
            ty_train.extend(ty_init)
            # initializing the acquisition vector
            acquisitions = deque(maxlen=training_size)
            acquisitions.extend(np.full(len(y_train), True))
            # train the model with the initially available data
            clf.fit(X_train, y_train)
            yhat = clf.predict(X_train)
            # initialize the list that stores the result of the classifier's prediction
            correct_classifications = [y_ == y for y_, y in zip(yhat, y_train)]
            init_acc = np.sum(correct_classifications) / len(yhat)
            print('Init class accuracy: {}'.format(init_acc))
            # # Initialized the drift detector
            err_sig = 1 - (np.asarray(correct_classifications).astype(int))
            for e in X_train:
                u_drift.add_element(e)
            for e in err_sig:
                s_drift.add_element(e)
            # initialize the number of acquired labels
            count = 0
            u_drift_idx = []
            s_drift_idx = []
            drift_flag = False
            # iterate over the whole data stream
            for t, (x_t, y_t, tX_t, ty_t) in enumerate(zip(X_stream, y_stream, tX_stream, ty_stream)):
                # infer the currently available labels
                # missing_label is used to denote unlabeled instances
                X_cand = x_t.reshape([1, -1])
                y_cand = y_t
                tX_cand = tX_t
                ty_cand = ty_t

                # manage delay in label. Do not forget. Do not add unavailable labels due delay.
                y_train_current = np.array(
                    [y if ty < tX_cand and a else missing_label for ty, y, a in zip(ty_train, y_train, acquisitions)])
                # # Semi-supervised drift detection
                if len(y_train_current) > verification_latency:
                    last_queried = y_train_current[-verification_latency - 1]
                    if last_queried != np.nan:
                        e = 1 - (np.asarray(correct_classifications)[-verification_latency - 1]).astype(int)
                        s_drift.add_element(e)
                    if s_drift.detected_change():
                        s_drift_idx.append(t)
                        drift_flag = True
                        # true_acquisitions = [ty < tX_cand and a for ty, a in zip(ty_train, acquisitions)]
                        # pending_acq = np.sum(np.asarray(true_acquisitions) == False)
                        # n = len(acquisitions)
                        # acquisitions.clear()
                        # acquisitions.extend([False for _ in range(n)])
                        # y_train_current = np.full(len(y_train_current), np.nan)
                    else:
                        drift_flag = False
                # Unsupervised drift
                u_drift.add_element(X_cand)
                if u_drift.detected_change():
                    u_drift_idx.append(t)
                    drift_flag = True
                else:
                    drift_flag = False

                # evaluation strategy: test-than-train
                # evaluate the prediction of the classifier
                yhat = clf.predict(X_cand)[0]
                correct_classifications.append(yhat == y_cand)
                # train the classifier
                if np.sum(~np.isnan(y_train_current)) > 0:
                    clf.fit(np.array(X_train), np.array(y_train_current))

                # check whether to sample the instance or not
                sampled_indices, utilities = call_func(delay_wrapper.query, X_cand=X_cand, clf=clf, X=np.array(X_train),
                                                       y=np.array(y_train_current),
                                                       tX=np.array(tX_train), ty=np.array(ty_train),
                                                       tX_cand=[tX_cand],
                                                       ty_cand=[ty_cand], return_utilities=True,
                                                       acquisitions=acquisitions,
                                                       drift=False)
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
                X_train.append(x_t)
                y_train.append(y_cand)
            # calculate and show the average accuracy
            print("Delay Wrapper: ", delay_wrapper_name, ", Avg Accuracy: ",
                  np.sum(correct_classifications) / stream_length, ", Number of acquired instances: ", count)
            # smoothed_curve = gaussian_filter1d(np.array(correct_classifications, dtype=float), 20)
            # smoothed_curve = np.array(correct_classifications)
            # smoothing the accuracy for plotting
            smoothing_window_length = 100
            s_idx = np.asarray(s_drift_idx)  # - smoothing_window_length + init_train_length + 1
            u_idx = np.asarray(u_drift_idx)  # - smoothing_window_length + init_train_length + 1
            # cumsum_correct_classifications = np.cumsum(correct_classifications)
            # smoothed_curve = (cumsum_correct_classifications[smoothing_window_length:] -
            #                   cumsum_correct_classifications[:-smoothing_window_length]) / smoothing_window_length
            # smoothed_curve = np.asarray(correct_classifications)
            smoothed_curve = np.convolve(correct_classifications, np.ones(smoothing_window_length),
                                         mode='valid') / smoothing_window_length  # len: max(M,N) - min(M,N) + 1
            plt.plot(smoothed_curve, label=delay_wrapper_name)
            try:
                plt.plot(s_idx, smoothed_curve[s_idx], 'x', c='red', alpha=1.0, label='Supervised Drift points')
                plt.plot(u_idx, smoothed_curve[u_idx], 'x', c='blue', alpha=1.0, label='Unsupervised Drift points')
            except:
                print(f'Drift not found')
            plt.title(f'{query_strategy_name}')
            if query_strategy_name == 'RandomSampler':
                break
        plt.legend()
    plt.show(block=True)
