import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import deque

from skactiveml.utils import call_func
from skactiveml.classifier import PWC
from skactiveml.stream import FixedUncertainty, VariableUncertainty, Split, PALS, RandomSampler, PeriodicSampler
from skactiveml.stream.budget_manager import FixedThresholdBudget, FixedUncertaintyBudget, BIQF, \
    VariableUncertaintyBudget, SplitBudget, EstimatedBudget
from skactiveml.stream.verification_latency import BaggingDelaySimulationWrapper, ForgettingWrapper, \
    FuzzyDelaySimulationWrapper
from skactiveml.classifier import SklearnClassifier


from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.switch_backend('Qt5Agg')


def get_randomseed(random_state):
    return random_state.randint(2 ** 31 - 1)


def linear_interp(a, b, alpha):
    return a * alpha + (1 - alpha) * b

######################################################################################################
if __name__ == '__main__':
    # random state that is used to generate random seeds
    random_state = np.random.RandomState(0)
    # number of instances that are provided to the classifier
    init_train_length = 10
    # the length of the data stream
    stream_length = 2000
    # the size of the sliding window that limits the training data
    training_size = 500
    # the verification latency occuring after querying a label
    verification_latency = 0
    budget = 0.1

    # Parameters for delay wrapper
    K = 2
    w_train = training_size
    delay_prior = 0.001

    # create the data stream
    # X, center = make_blobs(n_samples=init_train_length + stream_length, centers=30,
    #                       random_state=get_randomseed(random_state), shuffle=True)
    # y = center % 2
    #X, y = make_classification(n_samples=init_train_length + stream_length, n_features=2, n_informative=2,
    #                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0,
    #                           random_state=get_randomseed(random_state), shuffle=True)

    # Drifting Dataset
    X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=5.0,
                               return_centers=True)
    x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=2, centers=centers + 15,
                                               cluster_std=1.0,
                                               return_centers=True)
    # Gradual drift
    #for i in [x * 0.1 for x in range(1, 10, 1)]:
    #    x_, y_ = make_blobs(n_samples=300, n_features=2,
    #                        centers=linear_interp(centers_end, centers, i),
    #                        cluster_std=1.0)
    #    X = np.vstack([X, x_])
    #    y = np.hstack([y, y_])
    X = np.vstack([X, x_test_])
    y = np.hstack([y, Y_test_])
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
    clf_factory = lambda: PWC(classes=[0, 1], random_state=get_randomseed(random_state))
    #clf_factory = lambda: SklearnClassifier(SVC(probability=True), classes=[0, 1], random_state=get_randomseed(random_state))

    missing_label = clf_factory().missing_label
    query_strategies_factories = {
        'RandomSampler': lambda: RandomSampler(random_state=get_randomseed(random_state),
                                               budget_manager=FixedThresholdBudget(budget=budget)),
        # 'PeriodicSampler': lambda: PeriodicSampler(random_state=get_randomseed(random_state),
        #                                           budget_manager=FixedThresholdBudget(budget=budget)),
        'FixedUncertainty': lambda: FixedUncertainty(random_state=get_randomseed(random_state),
                                                    budget_manager=FixedUncertaintyBudget(budget=budget)),
        'VariableUncertainty': lambda: VariableUncertainty(random_state=get_randomseed(random_state),
                                                           budget_manager=VariableUncertaintyBudget(budget=budget)),
        'Split': lambda: Split(random_state=get_randomseed(random_state), budget_manager=SplitBudget(budget=budget)),
        'PALS': lambda: PALS(random_state=get_randomseed(random_state), budget_manager=BIQF(budget=budget))
    }

    for query_strategy_name, query_strategy_factory in query_strategies_factories.items():
        delay_wrappers_factories = {
            'None': lambda qs: qs,
            'Forgetting': lambda qs: ForgettingWrapper(base_query_strategy=qs, w_train=w_train,
                                                       random_state=get_randomseed(random_state)),
            #'BaggingDelaySimulation': lambda qs: BaggingDelaySimulationWrapper(base_query_strategy=qs,
            #                                                                   random_state=get_randomseed(
            #                                                                       random_state), K=K,
            #                                                                   delay_prior=delay_prior),
            # 'FuzzyDelaySimulation': lambda qs: FuzzyDelaySimulationWrapper(base_query_strategy=qs,
            #                                                               random_state=get_randomseed(random_state),
            #                                                               delay_prior=delay_prior)
        }
        # delay_wrappers_factories["Forgetting + Bagging"] = lambda qs: delay_wrappers_factories["Forgetting"](
        #     delay_wrappers_factories["BaggingDelaySimulation"](qs))
        # delay_wrappers_factories["Forgetting + Fuzzy"] = lambda qs: delay_wrappers_factories["Forgetting"](
        #    delay_wrappers_factories["FuzzyDelaySimulation"](qs))
        print("Query Strategy: ", query_strategy_name, )
        for delay_wrapper_name, delay_wrapper_factory in delay_wrappers_factories.items():
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
            # initialize the list that stores the result of the classifier's prediction
            correct_classifications = []
            # initialize the number of acquired labels
            count = 0
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
                # train the classifier
                clf.fit(np.array(X_train), np.array(y_train_current))
                # evaluate the prediction of the classifier
                correct_classifications.append(clf.predict(X_cand)[0] == y_cand)
                # check whether to sample the instance or not
                sampled_indices, utilities = call_func(delay_wrapper.query, X_cand=X_cand, clf=clf, X=np.array(X_train),
                                                       y=np.array(y_train_current),
                                                       tX=np.array(tX_train), ty=np.array(ty_train),
                                                       tX_cand=[tX_cand],
                                                       ty_cand=[ty_cand], return_utilities=True,
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
                X_train.append(x_t)
                y_train.append(y_cand)
            # calculate and show the average accuracy
            print("Delay Wrapper: ", delay_wrapper_name, ", Avg Accuracy: ",
                  np.sum(correct_classifications) / stream_length, ", Number of acquired instances: ", count)
            # smoothing the accuracy for plotting
            smoothing_window_length = 100
            cumsum_correct_classifications = np.cumsum(correct_classifications)
            plt.plot((cumsum_correct_classifications[smoothing_window_length:] - cumsum_correct_classifications[
                                                                                 :-smoothing_window_length]) / smoothing_window_length,
                     label=delay_wrapper_name)
            if query_strategy_name == 'RandomSampler':
                break
        plt.legend()
        plt.show(block=True)
