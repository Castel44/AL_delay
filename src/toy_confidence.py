import argparse
import json
import logging
import sys
import warnings
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from src.models.AEs import MLPAE
from src.models.MultiTaskClassification import AEandClass, LinClassifier, NonLinClassifier
from src.utils.induce_concept_drift import induce_drift, corrupt_drift
from src.utils.log_utils import StreamToLogger
from src.utils.montecarlo_dropout import get_monte_carlo_predictions, get_monte_carlo_embedding
from src.utils.new_dataload import OUTPATH
from src.utils.training_helper_v3 import *
from src.utils.utils import readable

from scipy.stats import entropy
from src.utils.torch_utils import predict_embedding
from scipy.special import softmax

from src.utils.montecarlo_dropout import get_monte_carlo_predictions, get_monte_carlo_embedding

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


def categorizer(y_cont, y_discrete):
    Yd = np.diff(y_cont, axis=0)
    Yd = (Yd > 0).astype(int).squeeze()
    C = pd.Series([x + y for x, y in
                   zip(list(y_discrete[1:].astype(int).astype(str)), list((Yd).astype(str)))]).astype(
        'category')
    return C.cat.codes


def map_abg_main(x):
    if x is None:
        return 'Variable'
    else:
        return '_'.join([str(int(j)) for j in x])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def check_2d3d(train, test, centroids):
    if train.shape[-1] > 3:
        from umap import UMAP
        trs = UMAP(n_components=2, n_neighbors=50, min_dist=0.01, metric='euclidean')
        train = trs.fit_transform(train)
        # valid = trs.transform(valid)
        test = trs.transform(test)
        centroids = trs.transform(centroids)
    return train, test, centroids


def plot_embedding_bary(train, test, Y_train, Y_test, centroids, classes, saver, figname='train_embedding'):
    print('Plot Embedding...')
    cmap = 'jet'
    COL = MplColorHelper(cmap, 0, classes)

    # train, test, centroids = check_2d3d(train, test, centroids)
    barycenters = centroids.mean(axis=1)

    plt.figure(figsize=(8, 6))
    if train.shape[1] == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    l0 = ax.scatter(*train.T, s=50, alpha=0.5, marker='.', label='Train',
                    c=COL.get_rgb(Y_train))
    l1 = ax.scatter(*test.T, s=50, alpha=0.5, marker='^', label='Test',
                    c=COL.get_rgb(Y_test))
    l2 = ax.scatter(*barycenters.T, s=250, marker='P', label='Barycenters',
                    c=COL.get_rgb([i for i in range(classes)]), edgecolors='black')
    for i in range(centroids.shape[0]):
        ax.scatter(*centroids[i].T, s=150, marker='x', label='Prototype',
                   c=COL.get_rgb([i]))
    lines = [l0, l1, l2] + [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=10) for c in
                            [COL.get_rgb(i) for i in np.unique(Y_train.astype(int))]]
    labels = [l0.get_label(), l1.get_label(), l2.get_label()] + [i for i in range(len(lines))]
    ax.legend(lines, labels)
    ax.set_title(figname)
    plt.tight_layout()
    saver.save_fig(plt.gcf(), figname)


def plot_embedding(train, test, Y_train, Y_test, centroids, classes, saver, figname='train_embedding'):
    print('Plot Embedding...')
    cmap = 'jet'
    COL = MplColorHelper(cmap, 0, classes)

    train, test, centroids = check_2d3d(train, test, centroids)

    plt.figure(figsize=(8, 6))
    if train.shape[1] == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    l0 = ax.scatter(*train.T, s=50, alpha=0.5, marker='.', label='Train',
                    c=COL.get_rgb(Y_train))
    l1 = ax.scatter(*test.T, s=50, alpha=0.5, marker='^', label='Test',
                    c=COL.get_rgb(Y_test))
    l2 = ax.scatter(*centroids.T, s=250, marker='P', label='Centroids',
                    c=COL.get_rgb([i for i in range(classes)]), edgecolors='black')
    lines = [l0, l1, l2] + [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=10) for c in
                            [COL.get_rgb(i) for i in np.unique(Y_train.astype(int))]]
    labels = [l0.get_label(), l1.get_label(), l2.get_label()] + [i for i in range(len(lines))]
    ax.legend(lines, labels)
    ax.set_title(figname)
    plt.tight_layout()
    saver.save_fig(plt.gcf(), figname)


# Model adaptation
def adapt_model(dl, model, loss_centroids, args):

    ######################################################################################################
    criterion = nn.CrossEntropyLoss(reduction='none', weight=None)

    print('Optimizer: ', args.optimizer)
    if 'SGD' in args.optimizer:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
            lr=args.learning_rate, weight_decay=args.l2penalty, momentum=args.momentum)
    else:
        optimizer = eval(args.optimizer)(
            list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
            lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = None
    # Train model
    model, loss_centroids = train_model(model, dl, dl, latent_constraint=loss_centroids,
                                        epochs=args.epochs, optimizer=optimizer,
                                        scheduler=scheduler, criterion=criterion,
                                        saver=saver, plot_loss_flag=args.plt_loss,
                                        clip=args.gradient_clip, abg=args.abg, sigma=args.sigma)
    print('Train ended')

    return model, loss_centroids


def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

    parser = argparse.ArgumentParser(description='Induced Concept Drift with benchmark data')

    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--informative', type=int, default=10)
    parser.add_argument('--redundant', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='blobs')
    parser.add_argument('--drift_features', type=str, default='top',
                        help='Which features to corrupt. Choices: top - bottom - list with features')
    parser.add_argument('--drift_type', type=str, default='gradual', help='flip or gradual')
    parser.add_argument('--drift_p', type=float, default=0.25, help='Percentage of features to corrupt')

    parser.add_argument('--weighted', action='store_true', default=False)

    parser.add_argument('--mu', type=float, default=0., help='Mu additive noise')
    parser.add_argument('--sigma', type=float, default=0., help='sigma additive noise')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--network', type=str, default='MLP',
                        help='Available networks: MLP')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--abg', type=float, nargs='+', default=[0, 1, 0])  # AE - Classification - Cluster
    parser.add_argument('--constr', type=str, default='centroids', choices=('centroids', 'multiple'))
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--optimizer', type=str, default='torch.optim.SGD')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--normalization', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--l2penalty', type=float, default=0.001)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=0, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number or runs')
    parser.add_argument('--process', type=int, default=1, help='Number of parallel process. Single GPU.')

    parser.add_argument('--nonlin_classifier', action='store_true', default=False, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=20)
    parser.add_argument('--embedding_size', type=int, default=2)

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[100, 100])

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_loss', action='store_true', default=True)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=True)
    parser.add_argument('--plt_cm', action='store_true', default=True)
    parser.add_argument('--plt_recons', action='store_true', default=True)
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
                                                                                                args.drift_type,
                                                                                                map_abg_main(args.abg)))

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
        cluster_std = 10
        if args.dataset == 'blobs':
            X, y, centers = make_blobs(n_samples=2000, n_features=n_features, centers=classes, cluster_std=cluster_std,
                                       return_centers=True)
            x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=n_features, centers=centers+25,
                                                       cluster_std=1.0,
                                                       return_centers=True)
            Y_test_ = Y_test_[:, np.newaxis]
            x_test, Y_test = make_blobs(n_samples=500, n_features=n_features, centers=centers, cluster_std=cluster_std)
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
            normalizer = eval(args.preprocessing)()
            x_train = normalizer.fit_transform(x_train)
            x_valid = normalizer.transform(x_valid)
            x_test = normalizer.transform(x_test)

        elif args.dataset.lower() == 'rbf':
            X, y = make_classification(n_samples=5000, n_features=n_features, n_informative=args.informative,
                                       n_redundant=args.redundant, n_repeated=0, n_classes=classes,
                                       n_clusters_per_class=2, weights=None, flip_y=0.0, class_sep=1., hypercube=True,
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
        #x_train = x_train[Y_train!=3]
        #Y_train = Y_train[Y_train!=3]
        #x_valid = x_valid[Y_valid!=3]
        #Y_valid = Y_valid[Y_valid!=3]

        # Encode labels
        unique_train = np.unique(Y_train)
        label_encoder = dict(zip(unique_train, range(len(unique_train))))
        unique_test = np.setdiff1d(np.unique(y), unique_train)
        label_encoder.update(dict(zip(unique_test, range(len(unique_train), len(np.unique(y))))))

        Y_train = np.array([label_encoder.get(e, e) for e in Y_train])
        Y_valid = np.array([label_encoder.get(e, e) for e in Y_valid])
        Y_test = np.array([label_encoder.get(e, e) for e in Y_test])

        # Weighted
        weights = None
        if args.weighted:
            print('Weighted')
            nSamples = np.unique(Y_train, return_counts=True)[1]
            tot_samples = len(Y_train)
            weights = (nSamples / tot_samples).max() / (nSamples / tot_samples)

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
        classes = len(np.unique(Y_train))
        args.nbins = classes
        history = x_train.shape[1]

        # Network definition
        if args.nonlin_classifier:
            classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim,
                                          dropout=args.dropout,
                                          norm=args.normalization)
        else:
            classifier = LinClassifier(args.embedding_size, classes)

        model_ae = MLPAE(input_shape=x_train.shape[1], embedding_dim=args.embedding_size, hidden_neurons=args.neurons,
                         hidd_act=eval(args.hidden_activation), dropout=args.dropout,
                         normalization=args.normalization).to(device)

        ######################################################################################################
        # model is multi task - AE Branch and Classification branch
        model = AEandClass(ae=model_ae, classifier=classifier, n_out=1, name='MLP').to(device)

        nParams = sum([p.nelement() for p in model.parameters()])
        s = 'MODEL: %s: Number of parameters: %s' % ('MLP', readable(nParams))
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

        ######################################################################################################
        if args.constr == 'centroids':
            loss_centroids = CentroidLoss(feat_dim=args.embedding_size, num_classes=classes, reduction='mean').to(
                device)
        elif args.constr == 'multiple':
            loss_centroids = PrototypeLoss(feat_dim=args.embedding_size, num_classes=classes, reduction='mean',
                                           n=args.n).to(device)
        else:
            raise ValueError(f'{args.constr} is not valid')

        criterion = nn.CrossEntropyLoss(reduction='none', weight=None)

        print('Optimizer: ', args.optimizer)
        if 'SGD' in args.optimizer:
            optimizer = eval(args.optimizer)(
                list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
                lr=args.learning_rate, weight_decay=args.l2penalty, momentum=args.momentum)
        else:
            optimizer = eval(args.optimizer)(
                list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
                lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = None
        # Train model
        model, loss_centroids = train_model(model, train_loader, valid_loader, latent_constraint=loss_centroids,
                                            epochs=args.epochs, optimizer=optimizer,
                                            scheduler=scheduler, criterion=criterion,
                                            saver=saver, plot_loss_flag=args.plt_loss,
                                            clip=args.gradient_clip, abg=args.abg, sigma=args.sigma)
        cluster_centers = loss_centroids.centers.detach().cpu().numpy()
        print('Train ended')

        ######################################################################################################
        # Eval
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=args.num_workers, pin_memory=True)
        train_results = evaluate_class_recons(model, x_train, Y_train, train_loader, saver,
                                              network=args.network, datatype='Train', plt_cm=args.plt_cm,
                                              plt_lables=False, plt_recons=True if args.abg[0] != 0 else False)
        valid_results = evaluate_class_recons(model, x_valid, Y_valid, valid_loader, saver,
                                              network=args.network, datatype='Valid', plt_cm=args.plt_cm,
                                              plt_lables=False, plt_recons=True if args.abg[0] != 0 else False)
        test_results = evaluate_class_recons(model, x_test, Y_test, test_loader, saver,
                                             network=args.network, datatype='Test', plt_cm=args.plt_cm,
                                             plt_lables=True, plt_recons=True if args.abg[0] != 0 else False)

        # Embeddings
        train_embedding = predict(model.encoder, train_loader).squeeze()
        valid_embedding = predict(model.encoder, valid_loader).squeeze()
        test_embedding = predict(model.encoder, test_loader).squeeze()

        # Get centroids when not constrained representation
        if args.abg[2] == 0:
            cc = []
            for i in range(len(np.unique(Y_train))):
                cc.append(train_embedding[Y_train == i].mean(axis=0))
            cluster_centers = np.array(cc)

        if train_embedding.shape[1] <= 3:
            try:
                plot_embedding(train_embedding, valid_embedding, Y_train, Y_valid, cluster_centers, classes=classes,
                               saver=saver,
                               figname='Train data')
                plot_embedding(valid_embedding, test_embedding[t_start:], Y_valid, Y_test[t_start:], cluster_centers,
                               classes=classes, saver=saver,
                               figname='Test (drift) data')
            except:
                plot_embedding_bary(train_embedding, valid_embedding, Y_train, Y_valid, cluster_centers,
                                    classes=classes,
                                    saver=saver,
                                    figname='Train data')
                plot_embedding_bary(valid_embedding, test_embedding[t_start:], Y_valid, Y_test[t_start:],
                                    cluster_centers,
                                    classes=classes, saver=saver,
                                    figname='Test (drift) data')
        else:
            print('Skipping embedding plot')

        plt.show(block=True)

        ############################################################################################################
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        x_pca_train = pca.fit_transform(x_train)
        x_pca_test = pca.transform(x_test)

        plt.figure()
        plt.scatter(*x_pca_train.T, c=Y_train)
        plt.title('Train')

        plt.figure()
        plt.scatter(*x_pca_test.T, c=Y_test)
        plt.title('Test')

        plt.show(block=True)

        ############################################################################################################
        #### Analysis prediction
        ############################################################################################################
        # Testing data
        def predict_analsys(model, loader, Y, classes=4, data='Train'):
            z, logit = predict_embedding(model, loader)
            yhat = np.argmax(softmax(logit, axis=1), axis=1)
            H = entropy(softmax(logit, axis=1), base=classes, axis=1)

            err = Y == yhat


            plt.figure()
            plt.scatter(*z.T, alpha=0.15, c='grey')
            plt.scatter(*z[~err].T, c='red', alpha=0.5, label='Error sample')
            #plt.scatter(*z[H>0.4].T, c='blue', alpha=0.5, label='High H')
            plt.legend()
            plt.title(f'{data}')

            plt.figure()
            f = plt.scatter(*z.T, c=H, cmap='seismic')
            plt.colorbar(f)
            plt.show(block=True)

        predict_analsys(model, train_loader, Y_train, classes, 'Train')


        # Monte Carlo Sampling
        data = train_loader
        labels = Y_train
        z, logit = predict_embedding(model, data)
        yhat = np.argmax(softmax(logit, axis=1), axis=1)
        H = entropy(softmax(logit, axis=1), base=classes, axis=1)
        err = labels == yhat

        MC_embedding = get_monte_carlo_embedding(data, model.encoder, args.embedding_size, forward_passes=50)
        MC = get_monte_carlo_predictions(data, model, classes, forward_passes=50)
        Y_mc = MC['mean'].argmax(axis=1)

        fig, axes = plt.subplots(nrows=5, sharex='all')
        axes[0].plot(MC['variance'].mean(axis=1), 'o-')
        axes[0].set_title('Mean Variance')
        axes[1].plot(MC['mean'].max(axis=1), 'o-')
        axes[1].set_title('Max Mean')
        for ax, s in zip(axes[2:], ['variance', 'entropy', 'mutual_info']):
            ax.plot(MC[s], 'o-')
            ax.set_title(s)
            # ax.legend()

        zz = np.reshape(MC_embedding, (-1, args.embedding_size))
        yy = np.tile(labels, 50)
        plt.figure()
        for i in range(classes):
            plt.scatter(*zz[yy==i].T, alpha=0.1)
        for i in range(classes):
            plt.scatter(*z[labels==i].T, alpha=0.5, marker='o', s=100)
        plt.scatter(*z[MC['mean'].max(axis=1)<0.75].T, marker='*', s=250, c='black', alpha=0.5)
        plt.scatter(*z[~err].T, marker='*', s=250, c='gold', alpha=0.5)

        plt.show(block=True)





"""        z, logits = predict_embedding(model, test_loader)
        yhat = softmax(logits, axis=1)
        H = entropy(yhat, base=classes, axis=1)
        pred = np.argmax(yhat, axis=1)
        yy = np.argmin(cdist(z, cluster_centers), axis=1)
        
        
        # Density estimation
        from scipy.spatial.distance import cdist, pdist
        data = train_embedding
        labels = Y_train
        k = 20

        # Offline - Get training/validation statistics
        dist = np.empty(data.shape[0])
        q_dist = dict.fromkeys(np.unique(labels))
        dict_data = dict.fromkeys(np.unique(labels))
        for i in np.unique(labels):
            dict_data[i] = data[labels==i]

        for i in q_dist.keys():
            q_dist[i] = dict.fromkeys(('low', 'mid', 'high'))

        for i in np.unique(labels):
            idx = np.where(labels==i)[0]
            pairwise_distances = cdist(data[idx], data[idx])
            kdist = np.take_along_axis(pairwise_distances, np.argpartition(pairwise_distances, k)[:, 1:k+1], axis=1)
            dist_ = np.median(kdist, axis=1)
            dist[idx] = dist_

            # Add IRQ
            q_dist[i]['low'], q_dist[i]['mid'], q_dist[i]['high'] = np.quantile(dist_, (0.01, .5, .99))

        df_dist = pd.DataFrame.from_dict(q_dist).T

        # Train data space
        plt.figure()
        c = plt.scatter(*data.T, c=dist, cmap='seismic')
        plt.colorbar(c)
        plt.show(block=True)

        ############################################################################################################
        # Testing data
        from scipy.stats import entropy
        from src.utils.torch_utils import predict_embedding
        from scipy.special import softmax
        z, logits = predict_embedding(model, test_loader)
        yhat = softmax(logits, axis=1)
        H = entropy(yhat, base=classes, axis=1)
        pred = np.argmax(yhat, axis=1)
        yy = np.argmin(cdist(z, cluster_centers), axis=1)

        plot_embedding(z, z, Y_test, Y_test, cluster_centers,
                       classes=classes, saver=saver, figname='True')
        plot_embedding(z, z, pred, pred, cluster_centers,
                       classes=classes, saver=saver, figname='Predict')
        plt.figure()
        plt.plot(H, '-o')
        plt.title('Test entropy')

        plt.figure()
        plt.plot(Y_test, '-o', label='GT', alpha=1)
        plt.plot(pred, '-o', label='yhat', alpha=0.25)
        plt.plot(yy, '-o', label='argmin', alpha=0.25)
        plt.plot(np.where(yy == pred, np.nan, -1), 'o', c='red', label='yhat =/= argmin')
        plt.plot(np.where(Y_test == pred, np.nan, -2), 'o', c='red', label='yhay =/= GT')
        plt.legend()
        plt.show(block=True)

        # Streaming part - Testing
        #ood = np.array([]).reshape(0,2)
        ood = []
        ood_label = []
        y_pseudo = []
        xx = []
        ood_flag = []
        idk = []

        for i, (x,l, x_orig) in enumerate(zip(z, logits, x_test)):
            pred = softmax(l, axis=0)
            H = entropy(pred, base=classes, axis=0)
            y = np.argmax(pred, axis=0)

            # Density in reference data. Compared to predicted class
            pairwise_distances = cdist(x[np.newaxis, :], dict_data[y])
            knn_idx = np.argpartition(pairwise_distances, k)[:, 1:k+1] #idx of closest neighbors reference data
            kdist = pairwise_distances.T[knn_idx].squeeze()
            dist = np.median(kdist)

            print(f'Sample: {i} - proba:{pred} - cls: {y} - H: {H:.3f} - d:{dist:.3f} - ')

            # TODO: use lists instead of empty array
            if dist <= q_dist[y]['mid'] + q_dist[y]['low']:
                # In distribution
                #print(f'Sample {i}: ID')
                dict_data[y] = np.concatenate((dict_data[y], x[np.newaxis,:]), axis=0)
                ood_flag.append(0)
                idk.append(-2)
            else:
                # Out of distribution
                #ood = np.concatenate((ood, x[np.newaxis,:]), axis=0)
                ood.append(x)
                xx.append(x_orig)
                ood_flag.append(2 if dist >= q_dist[y]['high'] else 1)
                ood_label.append(y)
                idk.append(-1)

                print(f'Sample {i}: OOD - Severity: {ood_flag[-1]}')

                if len(ood) < k+1:
                    # Buffer initialization
                    print(f'\t\tLoading OOD buffer {len(ood)}/{k+1}')
                    continue

                # ood buffer is loaded!
                ood_pd = cdist(x[np.newaxis, :], np.asarray(ood))
                knn_idx = np.argpartition(ood_pd, k)[:, 1:k + 1]
                ## 1: almost inliner. Get y_hat as pseudo-label
                ## 2: ood. cluster for new label.
                knn_label = np.round(np.average(np.asarray(ood_label)[knn_idx])).astype(int)
                idk[-1] = y if knn_label == 1 else -1
                #y_pseudo.append(y)
                #kdist = ood_pd.T[knn_idx].squeeze()
                #dist = np.median(kdist)

            if np.sum(np.asarray(ood_flag)==2) > 200:
                # stop streaming and adapt model
                break
        # Get new labels
        #labels_o = np.asarray([x for x in ood_label if x!=0])
        #data = ood[k:]
        #y = np.asarray(y_pseudo)
        #x_ = np.asarray(xx[k:])
        #plt.scatter(*data.T)
        #plt.show(block=True)

        #for i in np.unique(pred):
        #    idx = np.where(pred==i)[0]
        #    idx_data = np.where(labels==i)[0]
        #    pairwise_distances = cdist(z[idx], data[idx_data])
        #    kdist = np.take_along_axis(pairwise_distances, np.argpartition(pairwise_distances, k)[:, 1:k+1], axis=1)
        #    dist_ = np.median(kdist, axis=1)
        #    out_binary[idx] = np.where(dist_ > q_dist[i]['high'], 1, 0)

        n = len(ood_flag)
        data = np.asarray(z)[:n]
        c = {0: 'green', 1:'gold', 2:'red'}

        plt.figure()
        plt.scatter(*data.T, c=[c[i] for i in ood_flag])
        plt.show(block=True)

        plt.figure()
        for i in np.unique(idk):
            plt.scatter(*data[np.asarray(idk)==i].T, label=f'{i}')
        plt.legend()
        plt.show(block=True)





"""