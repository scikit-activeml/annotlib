import numpy as np
import copy

from annotlib.standard import StandardAnnot
from annotlib.utils import check_labelling_array, check_shape

from sklearn.base import is_classifier
from sklearn.utils import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class ClassifierBasedAnnot(StandardAnnot):
    """ClassifierBasedAnnot

    Annotators can be seen as human classifiers. Hence, we use classifiers based on machine learning techniques to
    represent these annotators. Given a data set comprising samples with their true labels, a classifier is trained on
    a subset of sample-label-pairs. Subsequently, this trained classifier is used as proxy of an annotator.
    As a result, the labels for a sample are provided by this classifier as well as the confidence scores which are
    the posterior probability estimates for the predicted class label.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true: array-like, shape (n_samples)
        True class labels of the given samples X.
    n_annotators: int
        Number of annotators who are simulated.
    classifiers: sklearn.base.ClassifierMixin | list of ClassifierMixin, shape (n_annotators)
        The classifiers parameter is either a single sklearn classifier supporting :py:method::predict_proba` or a list
        of such classifiers. If the parameter is not a list, each annotator is simulated on the same classification
        model, whereas if it is a list, the annotators may be simulated on different classifier types or even different
        parametrisations. The default classifiers parameter is a list of SVMs with the same parameters.
    train_ratios: 'one-hot' | 'equidistant' | array-like, shape (n_annotators, n_classes)
        The entry `train_ratios_[j, i]` indicates the ratio of samples of class i used for training the classifier
        of annotator j,  e.g. `train_ratios_[2,4]=0.3`: 30% of the samples for class 4 are used to train the classifier
        of annotator with the id 2.
    features: array-like, boolean, shape (n_annotators, n_features)
        This parameter is a boolean array indicating which features are considered for the training of an
        annotator's classifier, e.g. features_[a] = [0, 0, 1] means the classifier of annotator a is trained with
        the last of the three available features.
    confidence_noise: array-like, shape (n_annotators)
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
    y_true_: numpy.ndarray, shape (n_samples)
        The true class labels of the given samples.
    train_ratios_: numpy.ndarray, shape (n_annotators, n_classes)
        The entry `train_ratios_[j, i]` indicates the ratio of samples of class i used for training the classifier
        of annotator j,  e.g. `train_ratios_[2,4]=0.3`: 30% of the samples for class 4 are used to train the classifier
        of annotator 2.
    classifiers_: list of sklearn.base.ClassifierMixin, shape (n_annotators)
        The fitted classification models of the annotators.
    features_: array-numpy.ndarray, boolean, shape (n_annotators, n_features)
        This parameter is a boolean array indicating which features are considered for the training of an
        annotator's classifier, e.g. features_[a] = [0, 0, 1] means the classifier of annotator a is trained with
        the last of the three available features.
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
    >>> # simulate annotators on the iris data set
    >>> annotators = ClassifierBasedAnnot(X=X, y_true=y_true, classifiers=classifiers, n_annotators=3)
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
    """

    def __init__(self, X, y_true, classifiers=None, n_annotators=5, train_ratios='one-hot', features=None,
                 confidence_noise=None, random_state=None):
        # check shape of samples and labels
        self.X_, self.y_true_ = check_X_y(X, y_true)
        n_samples = np.size(self.X_, 0)
        n_features = np.size(self.X_, 1)

        # check and set number of annotators, query number and queried samples
        self._check_parameters(n_annotators, n_samples, confidence_noise, random_state)

        # initialise container for class labels and confidence scores
        self.Y_ = np.empty((n_samples, self.n_annotators_))
        self.C_ = np.empty((n_samples, self.n_annotators_))

        # transform class labels to interval [0, n_classes-1]
        le = LabelEncoder().fit(self.y_true_)
        y_transformed = le.transform(self.y_true_)
        y_unique = np.unique(y_transformed)
        n_classes = len(y_unique)

        # check classifier models
        if not isinstance(classifiers, list):
            clf = SVC(random_state=self.random_state_, probability=True) if classifiers is None else classifiers
            classifiers = [copy.deepcopy(clf) for _ in range(self.n_annotators())]
        for clf in classifiers:
            if len(classifiers) != self.n_annotators() or not is_classifier(clf) or getattr(clf, 'predict_proba',
                                                                                            None) is None:
                raise TypeError('The parameter `classifiers` must be a single sklearn classifier or a list of sklearn '
                                'classifiers supporting the method :py:method::`predict_proba`.')
        self.classifiers_ = classifiers

        # check shape and values of label_acc parameter
        self.train_ratios_ = train_ratios
        if isinstance(train_ratios, str):
            if train_ratios == 'one-hot':
                train_ratios = np.empty((self.n_annotators(), n_classes))
                class_indices = np.arange(0, n_classes)
                for a_idx in range(self.n_annotators()):
                    class_j = a_idx % n_classes
                    train_ratios[a_idx, class_j] = 1
                    train_ratios[a_idx, class_indices != class_j] = 0.2
            elif train_ratios == 'equidistant':
                train_ratios = np.fromfunction(lambda j, i: (j + 1) * (1 / (n_annotators)),
                                               (n_annotators, n_classes), dtype=int)
        self.train_ratios_ = check_labelling_array(train_ratios, (self.n_annotators(), n_classes), 'train_ratios')

        # check features flag array
        self.features_ = np.full((self.n_annotators(), n_features), True) if features is None else features
        self.features_ = check_array(self.features_)
        self.features_ = check_shape(self.features_, (self.n_annotators(), n_features), parameter_name='features')
        if self.features_.dtype != np.dtype('bool'):
            raise TypeError('The parameter `features` must be a boolean array.')

        # container for generated class labels and confidence scores of simulated annotators
        class_indices = [np.where(y_transformed == c)[0] for c in y_unique]

        # generate class labels depending on the clustering and the corresponding labelling accuracies
        for a_idx in range(n_annotators):
            train_size = [int(self.train_ratios_[a_idx, c] * len(class_indices[c]) + .5) for c in y_unique]
            train = [self.random_state_.choice(class_indices[c], size=train_size[c], replace=False) for c in
                     y_unique]
            train = np.hstack(train)
            X_train = self.X_[train]
            X_train = X_train[:, self.features_[a_idx]]
            y_train = y_transformed[train]
            self.classifiers_[a_idx] = self.classifiers_[a_idx].fit(X_train, y_train)
            y_predict = self.classifiers_[a_idx].predict(self.X_[:, self.features_[a_idx]])
            self.Y_[:, a_idx] = le.inverse_transform(y_predict)
            self.C_[:, a_idx] = np.max(self.classifiers_[a_idx].predict_proba(self.X_[:, self.features_[a_idx]]),
                                       axis=1)

        self._add_confidence_noise(probabilistic=True)
