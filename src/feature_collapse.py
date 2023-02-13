import argparse
import warnings
import shutil

import numpy as np
from cycler import cycler
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs, make_moons, make_classification

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn

from src.models.fc_resnet import MLP, MetaModel, FCResNet
from src.KDE_uncert.train_utils import train_model, predict, evaluate_model
from src.utils.utils import readable

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns
sns.set_style("whitegrid")


######################################################################################################

def linear_interp(a, b, alpha):
    return a * alpha + (1 - alpha) * b


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - value)).argmin()
    return array[idx]


def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='blobs', choices=['blobs', 'moons', 'rbf'])
    parser.add_argument('--network', type=str, default='ResNet',
                        help='Available networks: MLP or ResNet')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--optimizer', type=str, default='torch.optim.Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--normalization', type=str, default='spectral')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--l2penalty', type=float, default=0.0)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=42, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')

    parser.add_argument('--embedding_size', default=None, help='None or int')

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[100, 100, 100])
    # ResNet
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--features', type=int, default=128)

    # Suppress terminal out

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

    backend = 'Qt5Agg'
    print('Swtiching matplotlib backend to', backend)
    plt.switch_backend(backend)
    print()

    ######################################################################################################
    # Loading data'
    if args.dataset == 'blobs':
        X, y = make_blobs(n_samples=2000, n_features=2, centers=[[3, 3], [-3, 3]], cluster_std=0.5)
        X1, y1 = make_blobs(n_samples=2000, n_features=2, centers=[[-3, -3], [3, -3]], cluster_std=0.5)
        X = np.vstack([X, X1])
        y = np.concatenate((y, y1), axis=0)
    elif args.dataset == 'moons':
        X, y = make_moons(n_samples=500, noise=0.1)
    elif args.dataset == 'rbf':
        X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                                   n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=2, weights=None, flip_y=0.0, class_sep=5., hypercube=True,
                                   shift=0.0, scale=1.0, shuffle=True, random_state=args.init_seed)

    x_train, x_valid, Y_train, Y_valid = train_test_split(X, y, shuffle=True, test_size=0.2)
    x_test = x_valid
    Y_test = Y_valid

    data_uniform = np.random.uniform(-15, 15, (15000, 2))
    log_proba = multivariate_normal.logpdf(data_uniform, mean=[0, 0], cov=np.eye(2))
    idx = (np.linalg.norm(data_uniform - np.array([-8, 8]), axis=1)).argmin()

    ## Data Scaling
    # normalizer = MinMaxScaler()
    # x_train = normalizer.fit_transform(x_train)
    # x_valid = normalizer.transform(x_valid)
    # x_test = normalizer.transform(x_test)
    # data_uniform = normalizer.transform(data_uniform)

    classes = len(np.unique(Y_train))
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train.shape, [(Y_train == i).sum() for i in np.unique(Y_train)])
    print('Validation:', x_valid.shape, Y_valid.shape,
          [(Y_valid == i).sum() for i in np.unique(Y_valid)])
    print('Test:', x_test.shape, Y_test.shape,
          [(Y_test == i).sum() for i in np.unique(Y_test)])

    ######################################################################################################
    if args.network == 'MLP':
        model_encoder = MLP(input_dim=x_train.shape[1], hidden_neurons=args.neurons, normalization=args.normalization,
                            dropout=args.dropout, output_projection_dim=args.embedding_size,
                            activation=eval(args.hidden_activation)).to(device)
        if args.embedding_size is not None:
            input_class = args.embedding_size
        else:
            input_class = args.neurons[-1]
    elif args.network == 'ResNet':
        model_encoder = FCResNet(input_dim=x_train.shape[1], features=args.features, depth=args.depth,
                                 normalization=args.normalization,
                                 dropout=args.dropout, output_projection_dim=args.embedding_size,
                                 activation=eval(args.hidden_activation)).to(device)
        if args.embedding_size is not None:
            input_class = args.embedding_size
        else:
            input_class = args.features
    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel(encoder=model_encoder, in_class=input_class, out_class=classes).to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % (args.network, readable(nParams))
    print(s)

    ######################################################################################################
    # Main loop
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long())
    valid_dateset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dateset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dateset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dateset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)
    space_loader = DataLoader(TensorDataset(torch.from_numpy(data_uniform).float()), batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    ######################################################################################################
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=None)

    print('Optimizer: ', args.optimizer)
    if 'SGD' in args.optimizer:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=args.learning_rate, weight_decay=args.l2penalty, momentum=args.momentum)
    else:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # scheduler = None

    # Train model
    model = train_model(model, train_loader, valid_loader, epochs=args.epochs, optimizer=optimizer,
                        scheduler=scheduler, criterion=criterion, clip=args.gradient_clip)
    print('Train ended')

    ######################################################################################################
    # Classification results
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    train_results = evaluate_model(model, train_loader, Y_train, datatype='Train')
    valid_results = evaluate_model(model, valid_loader, Y_valid, datatype='Valid')
    test_results = evaluate_model(model, test_loader, Y_test, datatype='Test')

    ######################################################################################################
    # Embeddings
    train_embedding = predict(model.encoder, train_loader).squeeze()
    valid_embedding = predict(model.encoder, valid_loader).squeeze()
    test_embedding = predict(model.encoder, test_loader).squeeze()
    space_embedding = predict(model.encoder, space_loader)

    space_proba = predict(model, space_loader)
    space_yhat = space_proba.argmax(axis=1)

    if train_embedding.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        space_embedding = pca.fit_transform(space_embedding)
        train_embedding = pca.transform(train_embedding)

    from matplotlib import colors

    cmap = colors.ListedColormap(['black', 'red', 'blue', 'green'])

    color_cycler = (cycler(color=['r', 'black', 'blue', 'yellow', 'green']))

    fig, ax = plt.subplots(ncols=2, figsize=(19.2, 10.8))
    ax[0].scatter(*data_uniform.T, c=log_proba, cmap='inferno', s=10, alpha=0.8)
    ax[0].scatter(*x_train.T, c=Y_train, cmap=cmap)
    ax[0].scatter(*data_uniform[idx], marker='*', c='black', s=500)
    ax[0].set_title('Input Space')

    # ax[1].scatter(*space_embedding.T, c=space_yhat, s=20, alpha=0.5, cmap=cmap)
    ax[1].scatter(*space_embedding.T, c=log_proba, cmap='inferno', s=10, alpha=0.5)
    ax[1].scatter(*train_embedding.T, c=Y_train, cmap=cmap)
    ax[1].scatter(*space_embedding[idx], marker='*', c='black', s=500)
    ax[1].set_title('Feature Space. Norm: {}'.format(args.normalization))

    plt.show(block=True)

    from sklearn.svm import SVC
    svm = SVC(kernel='rbf').fit(x_train, Y_train)
    print(svm.score(x_train, Y_train))
    print(svm.score(x_test, Y_test))

    y_pred = svm.predict(x_test)
    from sklearn.metrics import classification_report
    #print(classification_report(Y_test, y_pred))