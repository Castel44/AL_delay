from src.stream_al.selection_strategies.base import BaseSelectionStrategy
from src.stream_al.util.split_training_data import split_training_data, get_A_n, get_selected_A_geq_n_tx, \
    LABEL_NOT_SELECTED

import numpy as np
from scipy.spatial.distance import cdist


def get_available_labels(tx_n, ty_n, XT_dict, YT_dict, TY_dict):
    TX = np.array(list(XT_dict.keys()))
    YT = np.array([YT_dict[tx] for tx in TX])
    TY = np.array([TY_dict[tx] for tx in TX])

    map_A_geq_n = TY < tx_n
    map_selected = YT != LABEL_NOT_SELECTED
    map_labeled_until_ty_n = TY < ty_n
    map_A_geq_n = np.logical_and(np.logical_and(map_A_geq_n, map_selected), map_labeled_until_ty_n)

    return TX[map_A_geq_n]


class PRWrapper(BaseSelectionStrategy):
    def __init__(self, random_state, base_selection_strategy, k, l=0.013):
        self.radom_state = random_state
        self.base_selection_strategy = base_selection_strategy
        self.k = k
        self.l = l

    def utility(self, X, clf, **kwargs):
        tx_n = kwargs['tx_n']
        ty_n = kwargs['ty_n']
        XT_dict = kwargs['XT_dict']
        YT_dict = kwargs['YT_dict']
        TY_dict = kwargs['TY_dict']

        selected_A_geq_n_tx = get_selected_A_geq_n_tx(tx_n, ty_n, XT_dict, YT_dict, TY_dict)
        selected_Y = get_available_labels(tx_n, ty_n, XT_dict, YT_dict, TY_dict)  # kwargs['Ly_n']
        unique = len(np.unique(selected_Y))

        # selected_A_geq_n_tx = np.setdiff1d(np.array(list(XT_dict.keys())), selected_Y)

        k = np.min([self.k, len(selected_Y) - 1])

        if len(selected_A_geq_n_tx) and len(selected_Y) > 1 and unique > 1:
            A_geq_n_X_selected = np.array([XT_dict[tx_i] for tx_i in selected_A_geq_n_tx]).reshape(
                len(selected_A_geq_n_tx), -1)

            # selected Y -> Lx_n, Ly_n
            Lx_n = kwargs['Lx_n']
            Ly_n = kwargs['Ly_n']
            # Lsw_n = kwargs['Lsw_n']
            # X_n = kwargs['X_n']
            y_type = Ly_n[0].dtype

            pairwise_distance = cdist(A_geq_n_X_selected, Lx_n)
            knn_idx = np.argpartition(pairwise_distance, k, axis=1)[:, :k]  # idx of closest neighbors reference data
            # TY_norm = selected_Y / selected_Y.max()
            TY_norm = np.exp(-(self.l * np.abs(selected_Y - selected_Y.max())) ** 2)

            knn_labels = Ly_n[knn_idx].astype(int)
            w_t = TY_norm[knn_idx]
            y_ = []
            for k, w in zip(knn_labels, w_t):
                count = np.bincount(k, weights=w)
                y_.append(np.argmax(count).astype(y_type))

            X = A_geq_n_X_selected
            # if len(not_selected_A_geq_n_tx):
            #     k = np.min([self.k, len(not_selected_A_geq_n_tx) - 1])
            #     A_geq_n_X_noy_selected = np.array([XT_dict[tx_i] for tx_i in not_selected_A_geq_n_tx]).reshape(
            #         len(not_selected_A_geq_n_tx), -1)
            #     pairwise_distance = cdist(A_geq_n_X_noy_selected, Lx_n)
            #     knn_idx = np.argpartition(pairwise_distance, k, axis=1)[:, :k]
            #     knn_labels = Ly_n[knn_idx].astype(int)
            #     w_t = TY_norm[knn_idx]
            #     for k, w in zip(knn_labels, w_t):
            #         count = np.bincount(k, weights=w)
            #         y_.append(np.argmax(count).astype(y_type))
            #     X = np.vstack([A_geq_n_X_selected, A_geq_n_X_noy_selected])

            y_ = np.array(y_)
            new_kwargs = kwargs.copy()
            new_kwargs['add_X'] = X
            new_kwargs['add_Y'] = y_
            new_kwargs['add_SW'] = np.ones(shape=[len(y_)])
            new_kwargs['modified_training_data'] = True
            utilities = self.base_selection_strategy.utility(X, clf, **new_kwargs)
            return utilities
        else:
            return self.base_selection_strategy.utility(X, clf, **kwargs)

    def reset():
        self.base_selection_strategy.reset()

    def partial_fit(self, X, sampled, **kwargs):
        self.base_selection_strategy.partial_fit(X, sampled, **kwargs)


    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    backend = 'Qt5Agg'
    plt.switch_backend(backend)
    plt.style.use("paper")
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    color_dict = {0: "#FF0000", 1: "#0000FF"}

    X_array = np.array([x for x in XT_dict.values()]).squeeze()
    X_labeled = Lx_n
    Y_labeled = Ly_n
    X_requested = A_geq_n_X_selected
    Y_requested = np.array([YT_dict[x] for x in selected_A_geq_n_tx])
    X_current = Lx_n[-1]

    # Visualize decision boundaries
    x_min, x_max = X_array[:, 0].min() - 0.5, X_array[:, 0].max() + 0.5
    y_min, y_max = X_array[:, 1].min() - 0.5, X_array[:, 1].max() + 0.5
    h = 0.05  # stepsize in mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    Z_border = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_border = Z_border.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.contourf(xx, yy, Z, 6, cmap=cm, alpha=0.25)
    cs = ax.contour(xx, yy, Z_border, 0, colors='grey', alpha=0.3, linestyles='dashed')
    ax.scatter(*X_array.T, c='grey', alpha=0.25, marker='.', label='Unlabeled')
    ax.scatter(*X_labeled.T, c=Y_labeled, cmap=cm_bright, edgecolors='k', alpha=0.3, label='Laebeled')
    ax.scatter(*X_current.T, c='black', marker='d', alpha=1., s=100, label='Current')
    ax.text(X_current[0]+0.2, X_current[1]-1.1, r"${x}_{{n}}$", fontsize=8)
    for i, (x, id, w, y) in enumerate(zip(X_requested, knn_idx, w_t, Y_requested)):
        ax.scatter(*X_labeled[id].T, c=Y_labeled[id], cmap=cm_bright, edgecolors='k', alpha=1, s=[max(30, 200*(x)) for x in w])
        ax.scatter(*x.T, marker='*', c='black', edgecolors=color_dict[y], s=150, alpha=1, label='Label requested')
        ax.text(x[0]+0.2, x[1]-1.1, r"$\hat{{x}}_{{{aaa}}}$".format(aaa=i), fontsize=8)
    ax.set_xlim(left=1.3, right=8.5)
    lines = [Line2D([0], [0], marker='.', linestyle='', color='grey'),
             tuple([Line2D([0], [0], marker='o', linestyle='', color=cm_bright(y), mec='k') for y in np.unique(Y_labeled)]),
             Line2D([0], [0], marker='*', linestyle='', color='black', ms=8),
             Line2D([0], [0], marker='d', linestyle='', color='black', ms=7)]
    labels = ['Unlabeled', 'Labeled', 'Label requested', 'Current']
    ax.legend(lines, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, loc=2)
    plt.show(block=True)
    
    fig.savefig("/home/castel/PycharmProjects/torchembedding/results/main_al_new/plots/feature_space/fig1.pdf")
    """

