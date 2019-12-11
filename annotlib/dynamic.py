import numpy as np

from annotlib.utils import check_indices, check_positive_integer
from annotlib.base import BaseAnnot

from sklearn.utils import check_random_state, column_or_1d, check_X_y


class DynamicAnnot(BaseAnnot):
    """DynamicAnnot

    An instance of this class emulates annotators with a dynamic labelling performance.
    Therefore, it requires learning rates which describe the progress of the labelling performance.
    Such a learning rate is either positive or negative.
    If it is positive, the corresponding annotator's labelling performance improves during the labelling process.
    In contrast, a negative learning rate results in a decreasing labelling performance.

    A very important remark is the fact that the labelling performance of adversarial and non-adversarial annotators is
    oppositional.
    A good labelling performance implies a high labelling accuracy for a non-adversarial annotator, whereas a good
    labelling performance of an adversarial annotator implies a low labelling accuracy.
    There is a option which defines whether an annotator is allowed to be adversarial.
    To realise the development of the labelling performance, the predicted label of an annotator is flipped with a
    probability depending on the state of the labelling progress which is represented by the number of queries.

    The flip probability is computed by :math:`p_{flip}(\mu_i, q_i) = min(|\mu_i| \cdot q_i, 1)`,
    where :math:`\mu` is the learning rate of an annotator :math:`a_i` and :math:`q_i` is the number of queries
    processed by the annotator :math:`a_i`.

    Parameters
    ----------
    annotator_model: BaseAnnotators
        An object of the type Annotators.
    y_unique: array-like, shape (n_classes)
        The array of available class labels.
    learning_rates: array-like, shape (n_annotators)
        A learning rate for each annotator. The default learning rate of an annotator is sampled from a uniform
        distribution :math:`U(-0.001,0.001)`.
    adversarial: boolean | array or list of booleans, shape (n_annotators)
        Flag, whether adversarial annotators are allowed. By default, this parameter is false, so that the
        non-adversarial annotators tend to make random guesses.
    random_state: None | int | instance of numpy.Random.RandomState
        The random_state is applied for generating the default learning rates and for flipping class labels.

    Attributes
    ----------
    annotator_model_: BaseAnnotators
        An object of the type Annotators.
    y_unique_: array-like, shape (n_classes)
        The array of available class labels.
    learning_rates_: numpy.ndarray, shape (n_annotators)
        A learning rate for each annotator. The default learning rate of an annotator is sampled from a uniform
        distribution :math:`U(-0.001,0.001)`.
    adversarial_: numpy.ndarray, shape (n_annotators)
        Flags specifying which annotators are allowed to adversarial. By default, this parameter is false for each
        annotator, so that the non-adversarial annotators makes only random guesses for originally wrong decisions.
    random_state_: None | int | instance of numpy.Random.RandomState
        The random_state is applied for generating the default learning rates and for flipping class labels.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from annotlib import ClassifierBasedAnnot
    >>> # load iris data set
    >>> X, y_true = load_iris(return_X_y=True)
    >>> y_unique = np.unique(y_true)
    >>> # simulate three annotators on the iris data set
    >>> annotator_model = ClassifierBasedAnnot(X=X, y_true=y_true, n_annotators=3)
    >>> # make the simulated annotators dynamic with default learning rates
    >>> dynamic_annotators = DynamicAnnot(annotator_model=annotator_model, y_unique=y_unique)
    >>> # query class labels of 2 samples from 3 annotators
    >>> dynamic_annotators.class_labels(X=X[0:2], y_true=y_true[0:2]).shape
    (2, 3)
    >>> # check query values
    >>> dynamic_annotators.n_queries()
    array([1, 1, 1])
    """

    def __init__(self, annotator_model, y_unique, learning_rates=None, adversarial=False, random_state=None):
        self.y_unique_ = column_or_1d(y_unique)

        if not isinstance(annotator_model, BaseAnnot):
            raise TypeError('The parameter `annotator_model` must be of the type `BaseAnnot`.')
        self.annotator_model_ = annotator_model

        if not isinstance(adversarial, (list, np.ndarray)):
            adversarial = [adversarial for _ in range(self.n_annotators())]
        adversarial = column_or_1d(adversarial)
        try:
            adversarial.astype(bool, casting='equiv')
        except Exception:
            raise TypeError('The parameter `adversarial` must be a boolean.')
        if len(adversarial) != self.n_annotators():
            raise ValueError('There must be a flag for each annotator in the parameter `adversarial`.')
        self.adversarial_ = adversarial

        self.random_state_ = check_random_state(random_state)

        if learning_rates is None:
            learning_rates = self.random_state_.uniform(-0.005, 0.005, self.annotator_model_.n_annotators())
        else:
            learning_rates = column_or_1d(learning_rates)
        if len(learning_rates) != self.n_annotators():
            raise ValueError('There must be a learning rate for each annotator in the parameter `learning_rates`.')
        self.learning_rates_ = learning_rates

    def n_annotators(self):
        """Method for computing the number of annotators.

        Returns
        -------
        n_annotators_: int
            Number of BaseAnnot.
        """
        return self.annotator_model_.n_annotators()

    def n_queries(self):
        """Method for computing the number of queries posed to an annotator.

        Returns
        -------
        n_queries_: numpy.ndarray, shape (n_annotators)
            An entry n_queries_[a] indicates how many queries annotator a has processed.
        """
        return self.annotator_model_.n_queries()

    def queried_samples(self):
        """Abstract method for returning the samples for which the annotators were queried to provide class labels.

        Returns
        -------
        X_queried_: numpy.ndarray, shape (n_annotators, n_samples, n_features)
            An entry X_queried_[a] represents the samples for which the annotator a was queried to provide class labels.
        """
        return self.annotator_model_.queried_samples()

    def class_labels(self, X, annotator_ids=None, query_value=1, **kwargs):
        """Method returning the class labels of the given samples.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples whose class labels are queried.
        y_true: array-like, shape (n_samples)
            The true class label of each given sample.
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
        # check parameters
        X, y_true = check_X_y(X, kwargs.get('y_true'))
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')
        query_value = check_positive_integer(query_value, 'query_value')
        prev_n_queries = np.copy(self.n_queries())
        Y = self.annotator_model_.class_labels(X=X, annotator_ids=annotator_ids, query_value=query_value, **kwargs)

        # ensures constant accuracy for given number of queries
        random_states = [np.random.RandomState(prev_n_queries[a_idx]) for a_idx in range(self.n_annotators())]

        if np.sum(prev_n_queries) > 0:
            # obtain class labels
            for x_idx in range(len(X)):
                Y_x = Y[x_idx, :]
                for a_idx in range(self.annotator_model_.n_annotators()):
                    flip_p = min(abs(self.learning_rates_[a_idx]) * prev_n_queries[a_idx], 1)
                    flip = random_states[a_idx].binomial(1, flip_p)
                    if self.learning_rates_[a_idx] < 0:
                        if y_true[x_idx] == Y_x[a_idx]:
                            if flip and not self.adversarial_[a_idx]:
                                Y_x[a_idx] = random_states[a_idx].choice(self.y_unique_)
                            if flip and self.adversarial_[a_idx]:
                                false_labels = self.y_unique_[self.y_unique_ != y_true[x_idx]]
                                Y_x[a_idx] = random_states[a_idx].choice(false_labels)
                        else:
                            if flip and not self.adversarial_[a_idx]:
                                Y_x[a_idx] = random_states[a_idx].choice(self.y_unique_)
                    elif self.learning_rates_[a_idx] > 0 and y_true[x_idx] != Y_x[a_idx]:
                        if flip:
                            Y_x[a_idx] = y_true[x_idx]
                Y[x_idx, :] = Y_x
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
        return self.annotator_model_.confidence_scores(X, annotator_ids, **kwargs)