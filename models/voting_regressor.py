from abc import abstractmethod
from sklearn.base import RegressorMixin, TransformerMixin, clone
from sklearn.utils import Bunch, Parallel, delayed
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
import numpy as np


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise ValueError(
                    "Underlying estimator {} does not support sample weights."
                        .format(estimator.__class__.__name__)
                ) from exc
            raise
    else:
        estimator.fit(X, y)
    return estimator


class _BaseVoting(_BaseComposition, TransformerMixin):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _required_parameters = ['estimators']

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimators))

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators"""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights)
                if est[1] not in (None, 'drop')]

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """
        common fit operations.
        """
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        names, clfs = zip(*self.estimators)
        self._validate_names(names)

        n_isnone = np.sum(
            [clf in (None, 'drop') for _, clf in self.estimators]
        )
        if n_isnone == len(self.estimators):
            raise ValueError(
                'All estimators are None or "drop". At least one is required!'
            )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y,
                                             sample_weight=sample_weight)
            for clf in clfs if clf not in (None, 'drop')
        )

        self.named_estimators_ = Bunch()
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e
        return self

    def set_params(self, **params):
        """ Setting the parameters for the ensemble estimator

        Valid parameter keys can be listed with get_params().

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ensemble estimator,
            the individual estimators of the ensemble estimator can also be
            set or replaced by setting them to None.

        Examples
        --------
        # In this example, the RandomForestClassifier is removed
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier()
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)]
        eclf.set_params(rf=None)
        """
        return self._set_params('estimators', **params)

    def get_params(self, deep=True):
        """ Get the parameters of the ensemble estimator

        Parameters
        ----------
        deep : bool
            Setting it to True gets the various estimators and the parameters
            of the estimators as well
        """
        return self._get_params('estimators', deep=deep)


# @plelievre: Adding this because the current version of sklearn is below 0.21
# and the voting regressor was added in 0.21
class VotingRegressor(_BaseVoting, RegressorMixin):
    """Prediction voting regressor for unfitted estimators.
    .. versionadded:: 0.21
    A voting regressor is an ensemble meta-estimator that fits base
    regressors each on the whole dataset. It, then, averages the individual
    predictions to form a final prediction.
    Read more in the :ref:`User Guide <voting_regressor>`.
    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``None`` or ``'drop'``
        using ``set_params``.
    weights : array-like, shape (n_regressors,), optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not `None`.
    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2)])
    >>> print(er.fit(X, y).predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]
    See also
    --------
    VotingClassifier: Soft Voting/Majority Rule classifier.
    """

    def __init__(self, estimators, weights=None, n_jobs=None):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        Returns
        -------
        self : object
        """
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, "estimators_")
        return np.average(self._predict(X), axis=1,
                          weights=self._weights_not_none)

    def transform(self, X):
        """Return predictions for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        predictions
            array-like of shape (n_samples, n_classifiers), being
            values predicted by each regressor.
        """
        check_is_fitted(self, 'estimators_')
        return self._predict(X)
