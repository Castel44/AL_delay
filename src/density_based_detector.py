import argparse
import json
import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.KDE_uncert.memory_buffers import IDBuffer, OODBuffer
from src.models.AEs import MLPAE
from src.models.MultiTaskClassification import AEandClass, LinClassifier, NonLinClassifier
from src.utils.induce_concept_drift import induce_drift, corrupt_drift
from src.utils.log_utils import StreamToLogger
from src.utils.new_dataload import OUTPATH
from src.utils.torch_utils import predict_embedding
from src.utils.training_helper_v3 import *
from src.utils.utils import readable
from src.denstream.DenStream import DenStream

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

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


def accuracy_rejection_curve(yhat, y_true, score):
    x = np.arange(1, 100)
    p = np.percentile(score, x)
    p = np.r_[0, p, 1]
    x = np.r_[0, x, 100]
    accs = []
    for i in p[::-1]:
        idx = np.where(score < i)[0]
        print(idx.size)
        if idx.size == 0:
            accs.append(1)
            continue
        accs.append(accuracy_score(yhat[idx], y_true[idx]))

    fig, ax = plt.subplots()
    plt.plot(x, accs, 'o-')
    return np.asarray(accs)


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
    parser.add_argument('--abg', type=float, nargs='+', default=[0, 1, 1])  # AE - Classification - Cluster
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
            x_test_, Y_test_, centers_end = make_blobs(n_samples=1000, n_features=n_features, centers=centers + 25,
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
        # Density estimation
        # Only use correct prediction data
        # TODO: reject option with unsure sample.
        dataloader = train_loader
        labels = Y_train

        z, logit = predict_embedding(model, dataloader)
        yhat = np.argmax(softmax(logit, axis=1), axis=1)
        H = entropy(softmax(logit, axis=1), base=classes, axis=1)
        err = labels == yhat

        #acc = accuracy_rejection_curve(yhat, labels, H)

        plt.figure()
        plt.scatter(*z.T, alpha=0.25, c='grey')
        plt.scatter(*z[~err].T, c='red', marker='*', alpha=0.5, s=200)
        c=plt.scatter(*z.T, c=H, alpha=0.75, cmap=plt.get_cmap('seismic') )
        plt.colorbar(c)
        plt.show(block=True)

        # H-based erroes
        data = z[err]
        labels = labels[err]
        input_data = x_train[err]

        #err = H[err]<0.5
        #data = data[err]
        #labels = labels[err]
        #input_data = input_data[err]
        k = 20

        id_buffer = IDBuffer(data, labels, input_data, classes, k)
        q_dist, irq = id_buffer.get_density_measure()
        # TODO: check input data
        dist = np.concatenate([id_buffer.data_dist[i] for i in range(classes)])
        data = np.concatenate([id_buffer.data_class[i] for i in range(classes)])
        df_dist = pd.DataFrame.from_dict(q_dist).T

        # Train data space
        plt.figure()
        plt.scatter(*train_embedding.T, alpha=0.5, s=20, c='black')
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

        # pal = sns.color_palette('tab10')
        plt.figure()
        plt.scatter(*train_embedding.T, c='grey', alpha=0.25, s=10)
        plt.scatter(*z.T, c=Y_test, alpha=0.25, cmap='tab10')

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

        ############################################################################################################
        # Streaming part - Testing
        from sklearn.cluster import DBSCAN

        clusterer = DenStream(lambd=0.1, eps=1.5, beta=0.5, mu=3)

        z, logits = predict_embedding(model, test_loader)
        yhat = softmax(logits, axis=1)
        H = entropy(yhat, base=classes, axis=1)
        pred = np.argmax(yhat, axis=1)
        yy = np.argmin(cdist(z, cluster_centers), axis=1)

        # ood = np.array([]).reshape(0,2)
        dict_data = id_buffer.data_class
        ood_buffer = OODBuffer()

        for i, (x, l, x_orig) in enumerate(zip(z, logits, x_test)):
            pred = softmax(l, axis=0)
            H = entropy(pred, base=classes, axis=0)
            y = np.argmax(pred, axis=0)

            # Density in reference data. Compared to predicted class
            dist = OODBuffer.knn_dist(x, dict_data[y])
            print(f'Sample: {i} - proba:{pred.max():.3f} - cls: {y} - H: {H:.3f} - d:{dist:.3f}')

            # If distance measure is low. Sample is ID. Update ID buffer
            if dist <= q_dist[y]['75'] + 1 * irq[y]:
                # In distribution
                # print(f'Sample {i}: ID')
                # TODO: save cluster statistics like micro-clusters.
                #id_buffer.update(x, y, x_orig)
                ood_buffer.update(x, pred, 0, i)

            else:
                # Out of distribution
                # low < d < med :-> warning. Not severe ood
                if dist >= q_dist[y]['75'] + 5* irq[y]:
                    # Severe OOD. Data in very low-density area
                    ood = 2
                else:
                    # dist between 1.5 IRQ and 3 IRQ. Almost OOD.
                    ood = 1
                ood_buffer.update(x, pred, ood, i)

                data = np.asarray(ood_buffer.data)
                clusterer.partial_fit([x])

                print(f'Sample {i}: OOD - Severity: {ood}')

                #if len(ood_buffer.ood_label) < k + 1:
                #    # Buffer initialization
                #    print(f'\t\tLoading OOD buffer {len(ood_buffer.ood_label)}/{k + 1}')
                #    # Do not do the rest.
                #    continue

                ## If more then k+1 ood samples are in the buffer already
                ## ood analysis
                #data = np.asarray(ood_buffer.data)
                #data = data[np.asarray(ood_buffer.ood_label) > 0]
                #db = DBSCAN(eps=np.max(df_dist.to_numpy()), min_samples=k).fit(data)
                #db_labels = db.labels_
                #n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                #n_noise = list(db_labels).count(-1)
                #print(f'DBSCAN: clusters:{n_clusters} - noise:{n_noise}')

                ## ood buffer is loaded!
                # ood_pd = cdist(x[np.newaxis, :], np.asarray(ood))
                # knn_idx = np.argpartition(ood_pd, k)[:, 1:k + 1]
                ### 1: almost inliner. Get y_hat as pseudo-label
                ### 2: ood. cluster for new label.
                # knn_label = np.round(np.average(np.asarray(ood_label)[knn_idx])).astype(int)
                # idk[-1] = y if knn_label == 1 else -1
                ##y_pseudo.append(y)
                ##kdist = ood_pd.T[knn_idx].squeeze()
                ##dist = np.median(kdist)

                ####

            ood_ = np.asarray(ood_buffer.ood_label)
            if np.sum(ood_[ood_> 1]) > 200:
                # stop streaming and adapt model
                break
        # Get new labels
        # labels_o = np.asarray([x for x in ood_label if x!=0])
        # data = ood[k:]
        # y = np.asarray(y_pseudo)
        # x_ = np.asarray(xx[k:])
        # plt.scatter(*data.T)
        # plt.show(block=True)

        # for i in np.unique(pred):
        #    idx = np.where(pred==i)[0]
        #    idx_data = np.where(labels==i)[0]
        #    pairwise_distances = cdist(z[idx], data[idx_data])
        #    kdist = np.take_along_axis(pairwise_distances, np.argpartition(pairwise_distances, k)[:, 1:k+1], axis=1)
        #    dist_ = np.median(kdist, axis=1)
        #    out_binary[idx] = np.where(dist_ > q_dist[i]['high'], 1, 0)

        # DenStream
        data = np.asarray(ood_buffer.data)
        db_labels = clusterer.fit_predict(data)
        cluster_center = []
        cluster_radius = []
        for cluster in clusterer.p_micro_clusters:
            #     print(cluster.center(), cluster.radius())
            cluster_center.append(cluster.center())
            cluster_radius.append(cluster.radius())
        cluster_center = np.array(cluster_center)
        cluster_radius = np.array(cluster_radius)
        print(cluster_center.shape, cluster_radius.shape)

        id_x, id_y = id_buffer.get_data()
        fig, ax = plt.subplots()
        ax.scatter(*id_x.T, c='grey', label='id', alpha=0.25, s=10)
        ax.scatter(*data.T, c=db_labels, marker='d', alpha=0.5, cmap=plt.get_cmap('tab10'))
        ax.scatter(*cluster_center.T, marker='*', s=100, c='red')
        for idx in range(len(cluster_center)):
            circle = plt.Circle((cluster_center[idx, 0], cluster_center[idx, 1]), cluster_radius[idx], color='r',
                                fill=False, lw=2)
            ax.add_artist(circle)

        # Plot dbscan results
        #id_x, id_y = id_buffer.get_data()
        #plt.figure()
        #plt.scatter(*id_x.T, c='grey', label='id', alpha=0.25, s=10)
        #plt.scatter(*data[db_labels != -1].T, c=db_labels[db_labels != -1], marker='d', alpha=0.75,
        #            cmap=plt.cm.get_cmap('tab10'))
        #plt.scatter(*data[db_labels == -1].T, c='black', marker='d', alpha=0.25)
        #plt.legend()
        #plt.show()

        """        
        # LOF
        clf = LocalOutlierFactor(n_neighbors=k)
        lof_score = clf.fit_predict(np.asarray(ood))
        X_scores = clf.negative_outlier_factor_
        plt.title("Local Outlier Factor (LOF)")
        plt.scatter(*id_x.T, c=id_y, label='id', alpha=0.25, s=10, cmap=plt.cm.get_cmap('tab10'))
        plt.scatter(*np.asarray(ood).T, color="k", s=3.0, label="Data points")
        # plot circles with radius proportional to the outlier scores
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(*np.asarray(ood).T, s=1000 * radius,
            edgecolors="r",
            facecolors="none",
            label="Outlier scores",
        )
        plt.show()
        """

        ood = np.asarray(ood_buffer.ood_label)
        n = len(ood)
        c = {0: 'green', 1: 'gold', 2: 'red'}

        data = np.asarray(ood_buffer.data)
        plt.figure()
        plt.scatter(*train_embedding.T, s=10, c='grey', alpha=0.25)
        plt.scatter(*data.T, c=[c[i] for i in ood], alpha=0.5)
        plt.show(block=True)

        ########
        H = entropy(ood_buffer.proba, axis=1, base=classes)
        plt.plot(H, alpha=0.25, color='black')
        plt.scatter(np.arange(len(H)), H, c=[c[i] for i in ood], alpha=0.5)

