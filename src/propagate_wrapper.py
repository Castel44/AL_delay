import numpy as np

from abc import abstractmethod
from copy import deepcopy
from sklearn.base import clone
from sklearn.utils import check_array, check_scalar, check_consistent_length
from skactiveml.base import SingleAnnotStreamBasedQueryStrategy, SkactivemlClassifier
from skactiveml.utils import (
    fit_if_not_fitted,
    check_type,
    check_random_state,
    call_func,
)
from skactiveml.stream.verification_latency._delay_wrapper import SingleAnnotStreamBasedQueryStrategyDelayWrapper

from scipy.spatial.distance import cdist


class PropagateLabelDelayWrapper(
    SingleAnnotStreamBasedQueryStrategyDelayWrapper
):

    def __init__(
            self,
            base_query_strategy=None,
            weighted = True,
            k=5,
            random_state=None,
    ):
        super().__init__(base_query_strategy, random_state)
        self.k = k
        self.weighted=weighted

    def query(
            self,
            X_cand,
            clf,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight=None,
            return_utilities=False,
            al_kwargs={},
            **kwargs
    ):
        """Ask the query strategy which instances in X_cand to acquire.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.

        Returns
        -------
        queried_indices : ndarray of shape (n_queried_instances,)
            The indices of instances in X_cand which should be queried, with
            0 <= n_queried_instances <= n_samples.

        utilities: ndarray of shape (n_samples,), optional
            The utilities based on the query strategy. Only provided if
            return_utilities is True.
        """
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = self._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
        )

        # Check if the classifier and its arguments are valid.
        check_type(clf, SkactivemlClassifier, "clf")

        clf_fitted = fit_if_not_fitted(
            clf, X, y, sample_weight, print_warning=False
        )

        # available lables
        n = np.sum(~np.isnan(y))
        available_labels = np.where(~np.isnan(y))[0]

        different_labels = len(np.unique(y[available_labels]))
        queried_delay = np.intersect1d(np.where(acquisitions)[0], np.where(np.isnan(y))[0])
        k = np.min([self.k, len(available_labels) - 1])
        y_type = y.dtype
        if different_labels > 1 and len(queried_delay) > 0:
            new_y = np.copy(y)
            new_ty = np.copy(ty)

            pairwise_distance = cdist(X[queried_delay], X[available_labels])
            knn_idx = np.argpartition(pairwise_distance, k, axis=1)[:, :k]  # idx of closest neighbors reference data
            d_max = pairwise_distance.max()
            d_min = pairwise_distance.min()
            normalized_distance = (pairwise_distance - d_min)/(d_max - d_min)
            normalized_ty = ty/ty.max()

            for j, queried_idx in enumerate(queried_delay):
                knn_labels = y[available_labels][knn_idx][j].astype(int)
                #unique_labels = np.unique(knn_labels)
                if self.weighted:
                    w_t = normalized_ty[knn_idx][j] # time weight. Higher is better
                    #w_d = 1 / (normalized_distance[j][knn_idx][j] + 1e-10)# distance weight.
                    weight = w_t #* w_d
                else:
                    weight = None
                weighted_count = np.bincount(knn_labels, weights=weight)
                l_ = np.argmax(weighted_count).astype(y_type)
                new_y[queried_idx] = l_
                new_ty[queried_idx] = tX_cand

            query_kwargs = {
                "X_cand": X_cand,
                "clf": clone(clf),
                "X": X,
                "y": new_y,
                "tX": tX,
                "ty": new_ty,
                "tX_cand": [tX_cand],
                "ty_cand": [ty_cand],
                "acquisitions": acquisitions,
                "sample_weight": sample_weight,
                "return_utilities": True,
            }
            base_query_strategy_is_delay_wrapper = isinstance(
                self.base_query_strategy_,
                SingleAnnotStreamBasedQueryStrategyDelayWrapper,
            )
            if base_query_strategy_is_delay_wrapper:
                query_kwargs["al_kwargs"] = al_kwargs
            else:
                query_kwargs.update(al_kwargs)

            queried_indices, utilities = call_func(
                self.base_query_strategy_.query, **query_kwargs
            )
        else:
            query_kwargs = {
                "X_cand": X_cand,
                "clf": clone(clf_fitted),
                "X": X,
                "y": y,
                "tX": tX,
                "ty": ty,
                "tX_cand": [tX_cand],
                "ty_cand": [ty_cand],
                "acquisitions": acquisitions,
                "sample_weight": sample_weight,
                "return_utilities": True,
            }
            base_query_strategy_is_delay_wrapper = isinstance(
                self.base_query_strategy_,
                SingleAnnotStreamBasedQueryStrategyDelayWrapper,
            )
            if base_query_strategy_is_delay_wrapper:
                query_kwargs["al_kwargs"] = al_kwargs
            else:
                query_kwargs.update(al_kwargs)
            queried_indices, utilities = call_func(
                self.base_query_strategy_.query, **query_kwargs
            )



        if return_utilities:
            return queried_indices, utilities
        else:
            return queried_indices


    def update(self, X_cand, queried, budget_manager_param_dict):
        """Updates the budget manager and the count for seen and queried
        instances

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which could be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.

        queried : array-like of shape (n_samples,)
            Indicates which instances from X_cand have been queried.

        kwargs : kwargs
            Optional kwargs for budget_manager and query_strategy.

        Returns
        -------
        self : BaggingDelaySimulationWrapper
            The BaggingDelaySimulationWrapper returns itself, after it is
            updated.
        """
        self._validate_base_query_strategy()
        self.base_query_strategy_.update(
            X_cand,
            queried,
            budget_manager_param_dict=budget_manager_param_dict,
        )
        return self


    def _validate_data(
            self,
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
            reset=True,
            **check_X_cand_params
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X_cand : {array-like, sparse matrix} of shape (n_samples, n_features)
            The instances which may be queried. Sparse matrices are accepted
            only if they are supported by the base query strategy.
        X : array-like of shape (n_samples, n_features)
            Input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Labels of the input samples 'X'. There may be missing labels.
        tX : array-like of shape (n_samples)
            Arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,) (default=None)
            Sample weights for X, used to fit the clf.
        return_utilities : bool, optional
            If true, also return the utilities based on the query strategy.
            The default is False.
        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
        **check_X_cand_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`.

        Returns
        -------
        X_cand: np.ndarray, shape (n_candidates, n_features)
            Checked candidate samples
        X : array-like of shape (n_samples, n_features)
            Checked input samples used to fit the classifier.
        y : array-like of shape (n_samples)
            Checked Labels of the input samples 'X'. There may be missing
            labels.
        tX : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X'
        ty : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y'
        tX_cand : array-like of shape (n_samples)
            Checked arrival time of the input samples 'X_cand'
        ty_cand : array-like of shape (n_samples)
            Checked arrival time of the Labels 'y_cand'
        acquisitions : array-like of shape (n_samples)
            List of arrived labels. True if Label arrived otherwise False
        sample_weight : array-like of shape (n_samples,)
            Checked sample weights for X
        return_utilities : bool,
            Checked boolean value of `return_utilities`.
        """
        (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        ) = super()._validate_data(
            X_cand=X_cand,
            X=X,
            y=y,
            tX=tX,
            ty=ty,
            tX_cand=tX_cand,
            ty_cand=ty_cand,
            acquisitions=acquisitions,
            sample_weight=sample_weight,
            return_utilities=return_utilities,
            reset=reset,
            **check_X_cand_params
        )

        return (
            X_cand,
            X,
            y,
            tX,
            ty,
            tX_cand,
            ty_cand,
            acquisitions,
            sample_weight,
            return_utilities,
        )
