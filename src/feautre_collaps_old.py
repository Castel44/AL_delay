import argparse
import json
import logging
import sys
import warnings

import pandas as pd
from cycler import cycler
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.models.AEs import MLPAE
from src.models.MultiTaskClassification import AEandClass, LinClassifier, NonLinClassifier
from src.utils.log_utils import StreamToLogger
from src.utils.new_dataload import OUTPATH
from src.utils.training_helper_v3 import *
from src.utils.utils import readable

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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - value)).argmin()
    return array[idx]


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


def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # List handling: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

    parser = argparse.ArgumentParser(description='Induced Concept Drift with benchmark data')

    parser.add_argument('--features', type=int, default=2)
    parser.add_argument('--classes', type=int, default=2)

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
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--gradient_clip', type=float, default=-1)

    parser.add_argument('--optimizer', type=str, default='torch.optim.SGD')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--normalization', type=str, default='none')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--l2penalty', type=float, default=0.0)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--init_seed', type=int, default=0, help='RNG seed. Typ. 42, 420, 1337, 0, 69.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number or runs')
    parser.add_argument('--process', type=int, default=1, help='Number of parallel process. Single GPU.')

    parser.add_argument('--nonlin_classifier', action='store_true', default=True, help='Final Classifier')
    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=2)

    # MLP
    parser.add_argument('--neurons', nargs='+', type=int, default=[128, 128, 128, 128])

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
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0], network=os.path.join(
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

        X, y, centers = make_blobs(n_samples=2000, n_features=n_features, centers=[[3, 3], [-3, -3]], cluster_std=0.5,
                                   return_centers=True)
        X, y = make_moons(n_samples=2000)

        x_train, x_valid, Y_train, Y_valid = train_test_split(X, y, shuffle=True, test_size=0.2)
        x_test = x_valid
        Y_test = Y_valid

        data_uniform = np.random.uniform(-15, 15, (15000, 2))
        log_proba = multivariate_normal.logpdf(data_uniform, mean=[0, 0], cov=np.eye(2))
        idx = (np.linalg.norm(data_uniform - np.array([-5,5]), axis=1)).argmin()

        # fig, ax = plt.subplots()
        # ax.scatter(*data_uniform.T, c=log_proba, cmap='inferno', s=10, alpha=0.8)
        # ax.scatter(*X.T, c=y)
        # ax.scatter(*data_uniform[idx], marker='*', c='black', s=500)
        # plt.show(block=True)

        ## Data Scaling
        #normalizer = MinMaxScaler()
        #x_train = normalizer.fit_transform(x_train)
        #x_valid = normalizer.transform(x_valid)
        #x_test = normalizer.transform(x_test)
        #data_uniform = normalizer.transform(data_uniform)

        # Weighted
        weights = None

        classes = len(np.unique(Y_train))

        ###########################
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
        space_loader = DataLoader(TensorDataset(torch.from_numpy(data_uniform).float()), batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

        ######################################################################################################
        loss_centroids = CentroidLoss(feat_dim=args.embedding_size, num_classes=classes, reduction='mean').to(device)
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
            plot_embedding(train_embedding, valid_embedding, Y_train, Y_valid, cluster_centers, classes=classes,
                           saver=saver,
                           figname='Train data')
            plot_embedding(valid_embedding, test_embedding, Y_valid, Y_test, cluster_centers,
                           classes=classes, saver=saver,
                           figname='Test (drift) data')
        else:
            print('Skipping embedding plot')

        # plt.show(block=True)

        space_embedding = predict(model.encoder, space_loader)
        _, space_proba = predict_multi(model, space_loader)
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
