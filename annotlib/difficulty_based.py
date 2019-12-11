import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from annotlib.standard import StandardAnnot
from annotlib.utils import check_positive_integer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d, check_X_y
from sklearn.model_selection import RepeatedKFold
from sklearn.base import is_classifier
from sklearn.svm import SVC

from scipy.special import entr


class DifficultyBasedAnnot(StandardAnnot):
    """DifficultyBasedAnnot

    This class implements a simulation technique aiming at quantifying the difficulty of a sample. The estimated
    difficulty is used in combination with an annotator labelling performance to compute the probability that the
    corresponding annotator labels the sample correctly.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true: array-like, shape (n_samples)
        True class labels of the given samples X.
    n_annotators: int
        Number of annotators who are simulated.
    classifiers: sklearn.base.ClassifierMixin | list of ClassifierMixin, shape (n_classifiers)
        The classifiers parameter is either a single sklearn classifier supporting :py:method::predict_proba` or a
        list of such classifiers. If the parameter is not a list, the simplicity scores are estimate by a single
        classifier, whereas if it is a list, the simplicity scores can be estimated by different classifier types or
        different parametrisations. The default classifiers parameter is a single SVM
    alphas: array-like, shape (n_annotators)
        The entry alphas[a_idx] indicates the annotator labelling performance, which is in the interval (-inf, inf).
        The following properties are valid:
        - alphas[a_idx] = 0: annotator with index a_idx makes random guesses,
        - alphas[a_idx] = inf: annotator with index a_idx is almost always right,
        - alphas[a_idx] = -inf: annotator with index a_idx is almost always wrong (adversarial).
    n_splits: int
        Number of folds of the cross-validation.
    n_repeats: int
        Number of repeats of the cross-validation.
    confidence_noise: array-like, shape (n_annotators)
        An entry of confidence_noise defines the interval from which the noise is uniformly drawn, e.g.
        confidence_noise[a] = 0.2 results in sampling n_samples times from U(-0.2, 0.2) and adding this noise
        to the confidence scores. Zero noise is the default value for each annotator.
    random_state: None | int | instance of :py:class:`numpy.random.RandomState`
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
        An entry n_queries_[a] indicates how many queries annotator a has processed.
    queried_flags_: numpy.ndarray, shape (n_samples, n_annotators)
        An entry queried_flags_[i, j] is a boolean indicating whether annotator a_i has provided a
        class label for sample x_j.
    y_true_: numpy.ndarray, shape (n_samples)
        The true class labels of the given samples.
    alphas_: array-like, shape (n_annotators)
        The entry alphas_[a_idx] indicates the annotator labelling performance, which is in the
        interval (-inf, inf). The following properties are valid:
        - alphas_[a_idx] = 0: annotator with index a_idx makes random guesses,
        - alphas_[a_idx] = inf: annotator with index a_idx is almost always right,
        - alphas_[a_idx] = -inf: annotator with index a_idx is almost always wrong (adversarial).
    betas_: array-like, shape (n_annotators)
        The entry betas_[x_idx] represents the simplicity score of sample X_[x_idx], where betas_[x_idx] is in the
        interval [0, inf):
        - betas_[x_idx] = 0: annotator with index a_idx makes random guesses,
        - betas_[x_idx] = inf: annotator with index a_idx is always right, if alphas_[a_idx] > 0
    n_splits_: int
        Number of folds of the cross-validation.
    n_repeats: int
        Number of repeats of the cross-validation.
    random_state_: None | int | numpy.random.RandomState
        The random state used for generating class labels of the annotators.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.svm import SVC
    >>> # load iris data set
    >>> X, y_true = load_iris(return_X_y=True)
    >>> # create list of SVM and Gaussian Process classifier
    >>> classifiers = [SVC(C=1, probability=True), SVC(C=3, probability=True), GaussianProcessClassifier()]
    >>> # set labelling performances of annotators
    >>> alphas = [-3, 0, 3]
    >>> # simulate annotators on the iris data set
    >>> annotators = DifficultyBasedAnnot(X=X, y_true=y_true, classifiers=classifiers, n_annotators=3, alphas=alphas)
    >>> # the number of annotators must be equal to the number of classifiers
    >>> annotators.n_annotators()
    3
    >>> # query class labels of 100 samples from annotators a_0, a_2
    >>> annotators.class_labels(X=X[0:100], y_true=y_true[0:100], annotator_ids=[0, 2], query_value=100).shape
    (100, 3)
    >>> # check query values
    >>> annotators.n_queries()
    array([100,   0, 100])
    >>> # query confidence scores of these 100 samples from annotators a_0, a_2
    >>> annotators.confidence_scores(X=X[0:100], y_true=y_true[0:100], annotator_ids=[0, 2]).shape
    (100, 3)
    >>> # query values are not affected by calling the confidence score method
    >>> annotators.n_queries()
    array([100,   0, 100])
    >>> # labelling performance of annotator a_0 is adversarial (worse than guessing)
    >>> annotators.labelling_performance(X=X, y_true=y_true)[0] < 1/len(np.unique(y_true))
    True
    """

    def __init__(self, X, y_true, classifiers=None, n_annotators=None, alphas=None, n_splits=5, n_repeats=10,
                 confidence_noise=None, random_state=None):
        # check shape of samples and labels
        self.X_, self.y_true_ = check_X_y(X, y_true)
        n_samples = len(self.X_)

        # check and set number of annotators, query number and queried samples
        n_annotators = 5 if n_annotators is None else n_annotators
        self._check_parameters(n_annotators, n_samples, confidence_noise, random_state)

        # check alpha scores
        self.alphas_ = np.linspace(0, 2, self.n_annotators()) if alphas is None else column_or_1d(alphas)
        if len(self.alphas_) != self.n_annotators():
                raise ValueError('The parameter `alphas` must contain a single labelling performance value for each'
                                 'annotator.')

        # create class labels and confidence scores container
        self.Y_ = np.empty((n_samples, self.n_annotators()))
        self.C_ = np.empty((n_samples, self.n_annotators()))

        # transform class labels to interval [0, n_classes-1]
        le = LabelEncoder().fit(self.y_true_)
        n_classes = len(le.classes_)
        y_transformed = le.transform(self.y_true_)

        # check classifier models
        if not isinstance(classifiers, list):
            clf = SVC(random_state=self.random_state_, probability=True) if classifiers is None else classifiers
            classifiers = [clf]
        for clf in classifiers:
            if not is_classifier(clf) or getattr(clf, 'predict_proba', None) is None:
                raise TypeError('The parameter `classifiers` must be a single sklearn classifier or a list of sklearn '
                                'classifiers supporting the method :py:method::`predict_proba`.')

        # check n_splits and n_repeats
        self.n_splits_, self.n_repeats_ = check_positive_integer(n_splits), check_positive_integer(n_repeats)

        # estimate simplicity scores (proxies of difficulties) of samples
        entropy_corr = np.zeros(n_samples)
        test_per_sample = np.zeros(n_samples)
        for classifier in classifiers:
            rkf = RepeatedKFold(n_splits=self.n_splits_, n_repeats=self.n_repeats_, random_state=random_state)
            for train_index, test_index in rkf.split(self.X_):
                classifier = classifier.fit(self.X_[train_index], self.y_true_[train_index])
                P = classifier.predict_proba(self.X_[test_index])
                E = np.sum(entr(P) / np.log(n_classes), axis=1)
                y_pred = classifier.predict(self.X_[test_index])
                entropy_corr[test_index] += (y_pred == self.y_true_[test_index]) * E
                entropy_corr[test_index] += (y_pred != self.y_true_[test_index])
                test_per_sample[test_index] += 1
        entropy_corr /= test_per_sample
        self.betas_ = np.divide(1, entropy_corr) - 1

        # compute confidence scores
        self.C_ = 1 / (1 + (n_classes - 1) * np.exp(-self.betas_.reshape(-1, 1) @ self.alphas_.reshape(1, -1)))

        # generate class labels
        for a in range(self.n_annotators_):
            for x in range(len(self.X_)):
                acc = self.C_[x, a]
                p = [(1 - acc) / (n_classes - 1)] * n_classes
                p[y_transformed[x]] = acc
                self.Y_[x, a] = le.inverse_transform(self.random_state_.choice(range(n_classes), p=p))

        # add confidence noise
        self._add_confidence_noise(probabilistic=True)

    def plot_annotators_labelling_probabilities(self, figsize=(5, 3), dpi=150, fontsize=7):
        """
        Creates a plot of the correct labelling probabilities for given labelling performances and estimated
        sample simplicity scores.

        Returns
        -------
        fig : matplotlib.figure.Figure object

        ax : matplotlib.axes.Axes.
        """
        colors = cm.rainbow(np.linspace(0, 1, self.n_annotators() + 1))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for a_idx in range(self.n_annotators()):
            ax.scatter(self.betas_, self.C_[:, a_idx], color=colors[a_idx].reshape(1, -1),
                       label=r'annotator $a_' + str(a_idx) + r'$: $\alpha_' + str(a_idx) + '=' + str(
                           self.alphas_[a_idx]) + '$', s=np.full(len(self.betas_), 5))
        ax.legend(loc='best', fancybox=False, framealpha=0.5, fontsize=fontsize)
        ax.set_xlabel(r'inverse difficulty scores of samples: $\beta_\mathbf{x}$', fontsize=fontsize)
        ax.set_ylabel(r'correct labelling probability: $p(y_\mathbf{x} | \alpha_i, \beta_\mathbf{x})$',
                      fontsize=fontsize)

        return fig, ax
