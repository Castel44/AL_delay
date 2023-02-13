import numpy as np

from sklearn.datasets import make_blobs, fetch_openml
from sklearn.preprocessing import LabelEncoder
from collections import deque
import pickle as pkl


def merge_streams(d0, d1, random_state):
    streams = [deque(d0[0]), deque(d1[0])]
    labels = [deque(d0[1]), deque(d1[1])]

    #     print('len(streams[0])', len(streams[0]))
    #     print('len(streams[1])', len(streams[1]))

    change_point = len(streams[0])

    w = 50

    new_stream = []
    new_labels = []
    while len(streams[0]) > 0 and len(streams[1]) > 0:
        stream0_proba = 1 / (1 + np.exp((len(new_stream) - change_point) / w))
        stream_index = 0 if random_state.random_sample() < stream0_proba else 1

        new_stream.append(streams[stream_index].popleft())
        new_labels.append(labels[stream_index].popleft())

    #     print('len(streams[0])', len(streams[0]))
    #     print('len(streams[1])', len(streams[1]))

    if len(streams[1]) > 0:
        #         print('case 0')
        new_stream = np.concatenate([new_stream, streams[1]], axis=0)
        new_labels = np.concatenate([new_labels, labels[1]], axis=0)
    else:
        #         print('case 1')
        new_stream = np.concatenate([new_stream, streams[0]], axis=0)
        new_labels = np.concatenate([new_labels, labels[0]], axis=0)

    return new_stream, new_labels


def make_synthetic_with_changes(random_state, lengths, n_classes, n_center_per_class, n_features):
    random_state = random_state

    static_streams = []
    for i, l in enumerate(lengths):
        X, y_center = make_blobs(n_samples=l, n_features=n_features, shuffle=True,
                                 centers=n_classes * n_center_per_class, random_state=i)
        y = y_center % n_classes
        static_streams.append((X, y))
    return merge_streams(static_streams[0], static_streams[1], random_state)


def check_splitfeature(X, feature_to_split):
    if len(np.unique(X[:, feature_to_split])) == 1:
        return False

    splitpoint = np.quantile(X[:, feature_to_split], 0.5)
    static_dataset_before_change = X[:, feature_to_split] <= splitpoint
    static_dataset_after_change = np.logical_not(static_dataset_before_change)

    #     print('feature_to_split', feature_to_split)
    #     print('len(np.unique(X[:, feature_to_split]))', len(np.unique(X[:, feature_to_split])))
    #     print(np.sum(static_dataset_before_change))
    #     print(np.sum(static_dataset_after_change))
    #     print(len(X))
    return np.sum(static_dataset_after_change) / len(X) > 0.40


def make_static_with_change(random_state, dataset_name):
    X, y = get_openml_dataset(dataset_name)
    X = X.values
    shuffled_indices = np.arange(len(X))
    random_state.shuffle(shuffled_indices)
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    num_features = X.shape[1]

    feature_to_split = random_state.randint(num_features)
    while not check_splitfeature(X, feature_to_split):
        feature_to_split = random_state.randint(num_features)

    splitpoint = np.quantile(X[:, feature_to_split], 0.5)
    static_dataset_before_change = X[:, feature_to_split] <= splitpoint
    static_dataset_after_change = X[:, feature_to_split] > splitpoint
    X_without_split_feature = X[:, np.arange(num_features) != feature_to_split]

    X_before = X_without_split_feature[static_dataset_before_change, :]
    y_before = y[static_dataset_before_change]
    X_after = X_without_split_feature[static_dataset_after_change, :]
    y_after = y[static_dataset_after_change]
    return merge_streams((X_before, y_before), (X_after, y_after), random_state)


def get_openml_id(name):
    name_id_dict = {
        'phoneme': 1489,
        'mfeat-morphological': 18,
        'segment': 36,
        'kin8nm': 807,
        'space_ga': 737,
        'electricity': 151
    }
    return name_id_dict[name]


def get_openml_dataset(dataset_name):
    data = fetch_openml(data_id=get_openml_id(dataset_name), data_home='datasets')
    le = LabelEncoder()
    X = data.data
    y = le.fit_transform(data.target)
    return X, y


def get_pickled_dataset(dataset_name):
    with open('datasets/' + dataset_name + '.pkl', 'rb') as f:
        data = pkl.load(f)
    return data['X'], data['y']


def get_1D_test_dataset(random_state):
    X, y_center = make_blobs(n_samples=4000, n_features=1, centers=4, shuffle=True, random_state=random_state)
    y = y_center % 2
    return X, y


def load_raw_dataset_by_name(random_seed, name):
    random_state = np.random.RandomState(random_seed)
    dataset_generation_functions = {}

    # synthetic dataset
    dataset_generation_functions['D0'] = lambda random_state: make_synthetic_with_changes(random_state, [2000, 2000], 2,
                                                                                          10, 2)
    dataset_generation_functions['D1'] = lambda random_state: make_synthetic_with_changes(random_state, [2000, 2000], 3,
                                                                                          10, 2)
    dataset_generation_functions['D2'] = lambda random_state: make_synthetic_with_changes(random_state, [2000, 2000], 4,
                                                                                          10, 2)

    # static dataset
    dataset_generation_functions['D3'] = lambda random_state: make_static_with_change(random_state, "phoneme")
    dataset_generation_functions['D4'] = lambda random_state: make_static_with_change(random_state,
                                                                                      'mfeat-morphological')
    dataset_generation_functions['D5'] = lambda random_state: make_static_with_change(random_state, "segment")
    dataset_generation_functions['D6'] = lambda random_state: make_static_with_change(random_state, "kin8nm")
    dataset_generation_functions['D7'] = lambda random_state: make_static_with_change(random_state, "space_ga")

    # stream dataset
    dataset_generation_functions['D8'] = lambda random_state: get_openml_dataset("electricity")
    dataset_generation_functions['D9'] = lambda random_state: get_pickled_dataset("luxembourg")
    dataset_generation_functions['D10'] = lambda random_state: get_pickled_dataset("noaa_weather")
    dataset_generation_functions['D11'] = lambda random_state: get_pickled_dataset("rialto")
    dataset_generation_functions['D12'] = lambda random_state: get_1D_test_dataset(random_state)

    # stream dataset
    dataset_generation_functions['D13'] = lambda random_state: make_synthetic_with_changes(random_state, [2000, 2000],
                                                                                           2, 1, 2)

    return dataset_generation_functions[name](random_state)


def create_eval_mask(random_seed, num_instances, eval_portion):
    num_eval_samples = int(num_instances * eval_portion)
    num_training_samples = num_instances - num_eval_samples
    #     print('num_eval_samples', num_eval_samples)
    #     print('num_training_samples', num_training_samples)
    eval_mask = np.concatenate([np.ones(num_eval_samples, dtype=int), np.zeros(num_training_samples, dtype=int)])
    np.random.RandomState(random_seed).shuffle(eval_mask)
    return eval_mask


def get_dataset_by_name(dataset_random_seed, evalset_random_seed, name):
    X, y = load_raw_dataset_by_name(dataset_random_seed, name)
    eval_mask = create_eval_mask(evalset_random_seed, len(X), 0.)
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    return X, y, eval_mask, n_classes, n_features


if  __name__ == '__main__':
    data = 'D9'

    x, y = load_raw_dataset_by_name(0, data)