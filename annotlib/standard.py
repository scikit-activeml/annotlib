import numpy as np

from annotlib.base import BaseAnnot
from annotlib.utils import check_indices, check_positive_integer, check_shape

from sklearn.utils import check_array, column_or_1d, check_random_state

from numpy_indexed import indices


class StandardAnnot(BaseAnnot):
    """StandardAnnot

    Standard annotators are represented by the class StandardAnnot, which enables to define
    arbitrary annotators. In a real-world scenario, an annotator is often a human who is asked to provide class labels
    for samples. An instance of the StandardAnnotators class aims at representing such a human within a
    Python environment.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    Y: array-like, shape (n_samples, n_annotators)
        Class labels of the given samples X.
    C: array-like, shape (n_samples)
        confidence score for labelling the given samples x.
    confidence_noise : array-like, shape (n_annotators)
        An entry of confidence_noise defines the interval from which the noise is uniformly drawn, e.g.
        confidence_noise[a] = 0.2 results in sampling n_samples times from U(-0.2, 0.2) and adding this noise
        to the confidence scores. Zero noise is the default value for each annotator.
    random_state: None | int | numpy.random.RandomState
            The random state used for generating class labels of the annotators.

    Attributes
    ----------
    X_: numpy.ndarray, shape (n_samples, n_features)
        Samples of the whole data set.
    Y_: numpy.ndarray, shape (n_samples, n_annotators)
        Class labels of the given samples X.
    C_: numpy.ndarray, shape (n_samples, n_annotators)
        confidence score for labelling the given samples x.
    C_noise_: numpy.ndarray, shape (n_samples, n_annotators)
        The uniformly noise for each annotator and each sample, e.g. C[x_idx, a_idx] indicates the noise for the
        confidence score of annotator with id a_idx in labelling sample with id x_idx.
    n_annotators_: int
        Number of annotators.
    n_queries_: numpy.ndarray, shape (n_annotators)
        An entry n_queries_[a_idx] indicates how many queries annotator with id a_idx has processed.
    queried_flags_: numpy.ndarray, shape (n_samples, n_annotators)
        An entry queried_flags_[a_idx, x_idx] is a boolean indicating whether annotator with id a_idx has provided a
        class label for sample with id x_idx.
    random_state_: None | int | numpy.random.RandomState
            The random state used for generating class labels of the annotators.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> X, y_true = load_iris(return_X_y=True)
    >>> # generate confidence scores
    >>> C = np.ones((len(X), 1))
    >>> # annotator is always correct
    >>> Y = y_true.reshape(-1,1)
    >>> annotator = StandardAnnot(X=X, Y=Y, C=C)
    >>> # number of annotators
    >>> print(annotator.n_annotators())
    1
    >>> # labelling performance of annotator
    >>> print(annotator.labelling_performance(X=X, y_true=y_true))
    [1.0]
    """

    def __init__(self, X, Y, C=None, confidence_noise=None, random_state=None, probabilistic=False):

        # check samples, class labels and confidence scores
        self.X_, self.Y_, = check_array(X), check_array(Y, force_all_finite=False)
        C = np.full(self.Y_.shape, np.nan) if C is None else C
        self.C_ = check_shape(check_array(C, force_all_finite=False), self.Y_.shape, 'C')
        if np.size(X, 0) != np.size(self.Y_, 0):
            raise ValueError('The number of samples and class labels must be equal.')

        # check remaining attributes
        self._check_parameters(np.size(Y, 1), np.size(X, 0), confidence_noise, random_state)
        self._add_confidence_noise(probabilistic)

    def n_annotators(self):
        """Method returning the number of annotators.

        Returns
        -------
        n_annotators: int
            Number of BaseAnnot.
        """
        return self.n_annotators_

    def n_queries(self):
        """Method returning the number of queries posed to an annotator.

        Returns
        -------
        n_queries_: numpy.ndarray, shape (n_annotators)
            An entry n_queries_[a] indicates how many queries annotator a has processed.
        """
        return self.n_queries_

    def queried_samples(self):
        """Method returning the samples for which the annotators were queried to provide class labels.

        Returns
        -------
        X_queried: list, shape (n_annotators, n_samples, n_features)
            An entry X_queried_[a] represents the samples for which the annotator a was queried to provide class labels.
        """
        return [self.X_[self.queried_flags_[:, a]] for a in range(self.n_annotators())]

    def class_labels(self, X, annotator_ids=None, query_value=1, **kwargs):
        """Method returning the class labels of the given samples.
        If the query value is greater than zero, it updates the n_queries and queried sample statistics

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples whose class labels are queried.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose class labels are queried.
        query_value: int
            The query value represents the increment of the query statistics of the queried annotators.

        Returns
        -------
        Y: numpy.ndarray, shape (n_samples, n_annotators)
            Class labels of the given samples which were provided by the queried annotators.
            The non queried annotators return np.nan values.
        """
        # check annotator_ids
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')

        # obtain ids of queried samples
        X = check_array(X)
        sample_ids = indices(self.X_, X, missing=-1)
        sample_ids_flag = sample_ids >= 0

        # class labels provided by queried annotators
        Y = np.full((np.size(X, 0), self.n_annotators()), np.nan)
        Y[sample_ids_flag, annotator_ids[:, None]] = self.Y_[sample_ids[sample_ids_flag], annotator_ids[:, None]]

        # update query statistics
        if query_value > 0:
            self.queried_flags_[sample_ids, annotator_ids[:, None]] = True
            self.n_queries_[annotator_ids] += query_value

        return Y

    def confidence_scores(self, X, annotator_ids=None, **kwargs):
        """Method returning the confidence scores for labelling the given samples.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples whose class labels are queried.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose confidence scores are queried.

        Returns
        -------
        C: numpy.ndarray, shape (n_samples, n_annotators)
            confidence scores of the queried annotators for labelling the given samples.
            The non queried annotators should return np.nan values.
        """
        # check annotator_ids
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')

        # obtain ids of queried samples
        X = check_array(X)
        sample_ids = indices(self.X_, X, missing=-1)
        sample_ids_flag = sample_ids >= 0

        # confidence scores provided by queried annotators
        C = np.full((np.size(X, 0), self.n_annotators()), np.nan)
        C[sample_ids_flag, annotator_ids[:, None]] = self.C_[sample_ids[sample_ids_flag], annotator_ids[:, None]]

        return C

    def _check_parameters(self, n_annotators, n_samples, confidence_noise, random_state):
        """
        This method is responsible for checking several parameters and to set them as attributes.

        Parameters
        ----------
        n_annotators: int
            Number of annotators.
        n_samples: int
            Number of samples.
        confidence_noise: array-like, shape (n_samples)
            Noise of the confidence scores of each annotator.
        random_state: None | int | instance of :py:class:`numpy.random.RandomState`
            The random state used for generating class labels of the annotators.
        """
        self.n_annotators_ = check_positive_integer(n_annotators, parameter_name='n_annotators')
        self.n_queries_ = column_or_1d(np.asarray([0] * self.n_annotators()))
        self.queried_flags_ = np.zeros((n_samples, n_annotators), dtype=bool)

        # check confidence noise
        self.confidence_noise_ = np.zeros(self.n_annotators()) if confidence_noise is None else confidence_noise
        self.confidence_noise_ = column_or_1d(self.confidence_noise_)

        if len(self.confidence_noise_) != self.n_annotators():
            raise ValueError('The number of elements in `confidence_noise` must be a equal to the number of annotators.')

        # check random state
        self.random_state_ = check_random_state(random_state)

        # add confidence noise
        self.C_noise_ = np.asarray(
            [self.random_state_.uniform(-self.confidence_noise_[a], self.confidence_noise_[a], n_samples) for a in
             range(self.n_annotators())]).T

    def _add_confidence_noise(self, probabilistic=False):
        """
        Add the uniform confidence noise to the confidence scores.

        Parameters
        ----------
        probabilistic: boolean
            If true, the confidence scores are in the interval [0, 1].
        """
        # adjust confidence values
        self.C_ += self.C_noise_
        if probabilistic:
            self.C_[self.C_ > 1] = 1
            self.C_[self.C_ < 0] = 0
