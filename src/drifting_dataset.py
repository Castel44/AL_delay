import numpy as np
from sklearn.datasets import make_blobs
from src.utils.concept_drift_datasets import check_datasets, load_data, AVAILABLE_DATA
from src.concept_drift.syth_data_creation import create_sea_drift_dataset, create_stagger_drift_dataset, \
    create_rotating_hyperplane_dataset, create_blinking_X_dataset
from skmultiflow.data import RandomRBFGeneratorDrift

from sklearn.preprocessing import StandardScaler
from src.utils.induce_concept_drift import  induce_drift, corrupt_drift


def load_induced_drift_dataset(dataset, p=0.5, type='flip'):
    X, y = load_data(dataset)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    stream_lenght = len(x_scaled)
    drift_pos = stream_lenght//2

    if type == 'flip':
        x_induced, permute_dict = induce_drift(x_scaled, y=y, t_start=drift_pos, t_end=None,
                                               p=p, features='top', copy=False)
    elif type == 'corrupt':
        x_induced, permute_dict = corrupt_drift(x_scaled, y=y, t_start=drift_pos, t_end=drift_pos+50,
                                                p=p, features='top', loc=3., std=2.0, copy=False)

    return x_induced, y, [drift_pos]


def create_random_rbf(n_samples_per_concept, n_classes=2, n_concept_drifts=2):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    gen = RandomRBFGeneratorDrift(n_classes=n_classes, n_centroids=10, change_speed=1.)
    gen.prepare_for_use()
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)

        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X)
        Y_stream.append(y)

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0)),
            "drifts": np.array(concept_drifts)}


def create_blobs(n_samples_per_concept, n_features=2, n_classes=2, n_concept_drifts=2):
    X, y = make_blobs(n_samples=n_samples_per_concept, n_features=n_features, centers=n_classes,
                               cluster_std=1., return_centers=False, shuffle=True)

    concept_drifts = []
    t = 0
    for _ in range(n_concept_drifts):
        X_, y_ = make_blobs(n_samples=n_samples_per_concept, n_features=n_features, centers=n_classes,
                                     cluster_std=1., return_centers=False, shuffle=True)
        t += n_samples_per_concept
        X = np.vstack([X, X_])
        y = np.hstack([y, y_])
        concept_drifts.append(t)

    return X, y, np.array(concept_drifts)


def create_synthetic_with_changes(n_samples_per_concept, n_classes=2, n_center_per_class=10, n_features=2,
                                  n_concept_drifts=2):
    X, y_center = make_blobs(n_samples=n_samples_per_concept, n_features=n_features, centers=n_classes*n_center_per_class,
                               cluster_std=1., shuffle=True, random_state=999)
    y = y_center % n_classes

    concept_drifts = []
    t = 0
    for i in range(n_concept_drifts):
        X_, y_center_, = make_blobs(n_samples=n_samples_per_concept, n_features=n_features, centers=n_classes*n_center_per_class,
                               cluster_std=1., shuffle=True, random_state=i)
        y_ = y_center_ % n_classes
        t += n_samples_per_concept
        X = np.vstack([X, X_])
        y = np.hstack([y, y_])
        concept_drifts.append(t)

    return X, y, np.array(concept_drifts)


def create_dataset(dataset_name, n_drifts=1, sample_per_concept=1000, **kwargs):
    if dataset_name.startswith("SYNTH"):
        n_class = int(dataset_name.split(sep='_')[2])
        n_features = int(dataset_name.split(sep='_')[1])
        X, y, drifts = create_synthetic_with_changes(sample_per_concept, n_classes=n_class, n_center_per_class=10,
                                                     n_concept_drifts=n_drifts, n_features=n_features)

    elif dataset_name.startswith("BLOBS"):
        n_class = int(dataset_name.split(sep='_')[2])
        n_features = int(dataset_name.split(sep='_')[1])
        X, y, drifts = create_blobs(sample_per_concept, n_features=n_features, n_classes=n_class,
                                    n_concept_drifts=n_drifts)

    elif dataset_name.startswith('fin_'):
        X, y, drifts = load_induced_drift_dataset(dataset_name, kwargs['p'], kwargs['type'])

    elif dataset_name in AVAILABLE_DATA:
        X, y = load_data(dataset_name)
        drifts = []

    else:
        if dataset_name.lower() == 'hyperplane':
            data_dict = create_rotating_hyperplane_dataset(sample_per_concept,
                                                           concepts=(np.pi/2*x for x in range(1,n_drifts+2)))

        elif dataset_name.lower() == 'stagger':
            data_dict = create_stagger_drift_dataset(sample_per_concept, n_concept_drifts=n_drifts+1)

        elif dataset_name.lower() == 'x':
            data_dict = create_blinking_X_dataset(sample_per_concept, n_concepts=n_drifts+1)

        elif dataset_name.lower() == 'sea':
            data_dict = create_sea_drift_dataset(sample_per_concept, n_concept_drifts=n_drifts+1)

        elif dataset_name.lower() == 'rbf':
            data_dict = create_random_rbf(sample_per_concept, n_concept_drifts=n_drifts+1)

        else:
            return ValueError('dataset unknown!')

        data = data_dict['data']
        drifts = data_dict['drifts']
        X, y = data

    return X, y, np.array(drifts)


if __name__ == '__main__':
    print(AVAILABLE_DATA.keys())
    name = 'hyperplane'

    X, y, drift = create_dataset(name, p=0.5, type='flip')
    print(X.shape)
    print(np.unique(y))