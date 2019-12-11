import numpy as np

from annotlib.utils import check_indices
from annotlib.base import BaseAnnot

from sklearn.utils import check_array

from itertools import chain


class MultiAnnotTypes(BaseAnnot):
    """MultiAnnotTypes

    This class enables to manage multiple types of annotators.

    Parameters
    ----------
    annotator_types: BaseAnnot | list, shape (n_annotators)
        A single annotator or a list of annotators who are to be added.

    Attributes
    ----------
    annotator_types_: list, shape (n_annotators)
        List of added annotators.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from annotlib import ClassifierBasedAnnot, ClusterBasedAnnot
    >>> X, y_true = load_iris(return_X_y=True)
    >>> # create two types of annotators
    >>> classifier_annotators = ClassifierBasedAnnot(X=X, y_true=y_true, n_annotators=3)
    >>> cluster_annotators = ClusterBasedAnnot(X=X, y_true=y_true, n_annotators=3)
    >>> annotator_types = [classifier_annotators, cluster_annotators]
    >>> # create instance of multiple annotator types
    >>> multiple_annotators = MultiAnnotTypes(annotator_types=annotator_types)
    >>> # there are 3+3=6 annotators
    >>> multiple_annotators.n_annotators()
    6
    >>> # ask 6 annotators for class labels of 10 samples
    >>> multiple_annotators.class_labels(X=X[0:10], query_value=10).shape
    (10, 6)
    >>> # check query values
    >>> multiple_annotators.n_queries()
    array([10, 10, 10, 10, 10, 10])
    """

    def __init__(self, annotator_types):
        annotator_types = [annotator_types] if isinstance(annotator_types, BaseAnnot) else annotator_types
        if isinstance(annotator_types, list):
            for a in annotator_types:
                if not isinstance(a, BaseAnnot):
                    raise TypeError(
                        'An annotator is required to be an instance of '
                        ':py:class:`annotlib.base.BaseAnnot`')
        else:
            raise TypeError('The parameter `annotator_types` must be a single annotator or a list of annotators.')
        self.annotator_types_ = annotator_types

    def add_annotators(self, annotator_types):
        """Method adds new annotators.

        Parameters
        ----------
        annotator_types: list, shape (n_annotators)
            The annotator types to be added.

        Returns
        -------
        self: sim_annotator_lib.multiple_annotator_types.MultiAnnotTypes
            The instance itself.
        """
        annotator_types = [annotator_types] if isinstance(annotator_types, BaseAnnot) else annotator_types
        for a in annotator_types:
            if not isinstance(a, BaseAnnot):
                raise TypeError(
                    'An annotator is required to be an instance of '
                    ':py:class:`annotlib.base.BaseAnnot`')
            else:
                self.annotator_types_.append(a)
        return self

    def n_annotators(self):
        """Method for computing the number of annotators.

        Returns
        -------
        n_annotators: int
            Number of annotators.
        """
        return np.sum([a.n_annotators() for a in self.annotator_types_])

    def n_queries(self):
        """Method for computing the number of queries posed to an annotator.

        Returns
        -------
        n_queries: numpy.ndarray, shape (n_annotators)
            An entry n_queries_[a] indicates how many queries annotator a has processed.
        """
        return np.hstack([a.n_queries() for a in self.annotator_types_])

    def queried_samples(self):
        """Abstract method for returning the samples for which the annotators were queried to provide class labels.

        Returns
        -------
        X_queried: numpy.ndarray, shape (n_annotators, n_queried_samples, n_features)
            An entry X_queried_[a] represents the samples for which the annotator a was queried to provide class labels.
        """
        return list(chain(*[a.queried_samples() for a in self.annotator_types_]))

    def class_labels(self, X, annotator_ids=None, query_value=1, **kwargs):
        """Method returning the class labels of the given samples.

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
        X = check_array(X)
        if annotator_ids is None:
            Y = np.hstack([a.class_labels(X, None, query_value, **kwargs) for a in self.annotator_types_])
        else:
            annotator_ids = self._transform_ids(check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids'))
            Y = []
            for a in range(len(self.annotator_types_)):
                if len(annotator_ids[a]) > 0:
                    Y_a = self.annotator_types_[a].class_labels(X=X, annotator_ids=annotator_ids[a],
                                                                query_value=query_value, **kwargs)
                else:
                    Y_a = np.empty((len(X), self.annotator_types_[a].n_annotators()))
                    Y_a.fill(np.nan)
                Y.append(Y_a)
            Y = np.hstack(Y)
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
            Confidence scores of the queried annotators for labelling the given samples.
            The non queried annotators should return np.nan values.
        """
        X = check_array(X)
        if annotator_ids is None:
            C = np.hstack([a.confidence_scores(X) for a in self.annotator_types_])
        else:
            annotator_ids = self._transform_ids(check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids'))
            C = []
            for a in range(len(self.annotator_types_)):
                if len(annotator_ids[a]) > 0:
                    C_a = self.annotator_types_[a].confidence_scores(X, annotator_ids[a])
                else:
                    C_a = np.empty((len(X), self.annotator_types_[a].n_annotators()))
                    C_a.fill(np.nan)
                C.append(C_a)
            C = np.hstack(C)
        return C

    def _transform_ids(self, annotator_ids):
        """
        This method transforms the annotator ids, so that the correct annotators in the different
        types are asked. If we have two annotator types each with two annotators, the ids [0, 1, 2, 3] are transformed
        to [[0, 1], [0, 1]], so that for each type there is a separate list of annotator ids.

        Parameters
        ----------
        annotator_ids: array-like,
            Annotator ids to be transformed.

        Returns
        -------
        new_ids: list, shape (n_annotator_types, n_asked_annotators_of_annotator_type)
            The transformed annotator ids. For example, the entry new_ids[1]=[0, 2] defines the annotators with the
            indices 0 and 2 of the annotator type with index 1 are asked.
        """
        id_ranges = np.zeros((len(self.annotator_types_), 2), dtype=int)
        id_ranges[0] = np.asarray([0, self.annotator_types_[0].n_annotators() - 1])
        for a in range(1, len(self.annotator_types_)):
            x_1 = id_ranges[a - 1][1] + 1
            x_2 = x_1 + self.annotator_types_[a].n_annotators() - 1
            id_ranges[a] = np.asarray([x_1, x_2], dtype=int)

        new_ids = list()
        for a in range(len(id_ranges)):
            new_ids.append([])
            for i in annotator_ids:
                if id_ranges[a, 0] <= i <= id_ranges[a, 1]:
                    new_ids[a].append(i - id_ranges[a, 0])
        return new_ids
