import numpy as np

from annotlib.standard import StandardAnnot
from annotlib.utils import check_labelling_array

from sklearn.utils import check_X_y, column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class ClusterBasedAnnot(StandardAnnot):
    """ClusterBasedAnnot

    A popular approach to simulate annotators is based on a clustering. The intention of such a clustering is the
    emulation of areas of knowledge. The assumption is that the knowledge of an annotator is not constant for a
    whole classification problem. Hence, there are areas where the annotator has a wider knowledge, whereas there
    are also areas for which an annotator has only a sparse knowledge. With reference to a classification problem,
    such an area of knowledge may be viewed as an area in the feature space or an area in the class label space.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Samples of the whole data set.
    y_true: array-like, shape (n_samples)
        True class labels of the given samples X.
    y_cluster: array-like, shape (n_samples)
        The cluster labels of each sample.
        Default is None, so that the class labels y_true are assumed as cluster labels.
    n_annotators: int
        Number of annotators who are simulated.
        Default is None, so that the number of annotators is equal to the number of clusters.
    cluster_labelling_acc: 'one-hot' | 'equidistant' | array-like, shape (n_annotators, n_clusters, 2)
        The interval of the labelling accuracy of each annotator for each cluster.
        From this interval, the labelling accuracy is drawn for each sample, e.g.
        cluster_label_acc[a_idx, c_idx]=[0.4, 0.6] means that the labelling accuracy of annotator with the id a_idx
        is sampled independently from U(0.4, 0.6) for each sample of cluster with index c_idx.
        The indices of the clusters are according to the numerical order for numerical cluster labels and according
        to the lexicographical order fo strings as cluster labels.
        The default value of this parameter is 'one-hot', so that each annotator is expert on a single cluster.
        Another option is 'equidistant', so that an annotator has the equal labelling performance on all clusters
        and the performance difference between the ith and (i+1)th best annotator is equidistant for all ranks i.
    confidence_noise: array-like, shape (n_annotators)
        An entry of confidence_noise defines the interval from which the noise is uniformly drawn, e.g.
        confidence_noise[a] = 0.2 results in sampling n_samples times from U(-0.2, 0.2) and adding this noise
        to the confidence scores.
        Zero noise is the default value for each annotator.
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
    cluster_labelling_acc_: float, shape (n_annotators, n_clusters, 2)
        The interval of the labelling accuracy of each annotator for each cluster.
        From this interval, the labelling accuracy is drawn for each sample, e.g. cluster_label_acc[a, c]=[0.4, 0.6]
        means that the labelling accuracy of annotator a is sampled independently from U(0.4, 0.6) for each sample
        of cluster c.
    y_cluster_: array-like, shape (n_samples)
        The cluster labels of each sample.
    random_state_: None | int | numpy.random.RandomState
            The random state used for generating class labels of the annotators.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.cluster import KMeans
    >>> # load iris data set
    >>> X, y_true = load_iris(return_X_y=True)
    >>> # cluster data with k-means (five clusters)
    >>> y_cluster = KMeans(n_clusters=5).fit_predict(X=X)
    >>> # simulate annotators on the clustered iris data set
    >>> annotators = ClusterBasedAnnot(X=X, y_true=y_true, y_cluster=y_cluster)
    >>> # by default the number of annotator is equal to the number of clusters
    >>> annotators.n_annotators()
    5
    >>> # query class labels of 100 samples from annotators a_0, a_2, a_4
    >>> annotators.class_labels(X=X[0:100], y_true=y_true[0:100], annotator_ids=[0, 2, 4], query_value=100).shape
    (100, 5)
    >>> # check query values
    >>> annotators.n_queries()
    array([100,   0, 100,   0, 100])
    >>> # query confidence scores of these 100 samples from annotators a_0, a_2, a_4
    >>> annotators.confidence_scores(X=X[0:100], y_true=y_true[0:100], annotator_ids=[0, 2, 4]).shape
    (100, 5)
    >>> # query values are not affected by calling the confidence score method
    >>> annotators.n_queries()
    array([100,   0, 100,   0, 100])
    """

    def __init__(self, X, y_true, y_cluster=None, n_annotators=None, cluster_labelling_acc='one_hot',
                 confidence_noise=None, random_state=None):
        # check samples and true class labels
        self.X_, self.y_true_ = check_X_y(X, y_true)
        n_samples = np.size(self.X_, 0)

        # check cluster labels
        y_cluster = self.y_true_ if y_cluster is None else y_cluster
        _, self.y_cluster_ = check_X_y(X, y_cluster)

        # transform class labels to interval [0, n_classes-1]
        le_cla = LabelEncoder().fit(self.y_true_)
        n_classes = len(le_cla.classes_)
        y_true_transformed = le_cla.transform(self.y_true_)

        # transform cluster labels to interval [0, n_clusters-1]
        le_clu = LabelEncoder().fit(self.y_cluster_)
        n_clusters = len(le_clu.classes_)
        y_cluster_transformed = le_clu.transform(self.y_cluster_)

        # check number of annotators, confidence noise, and random state
        n_annotators = n_clusters if n_annotators is None else n_annotators

        self._check_parameters(n_annotators, n_samples, confidence_noise, random_state)
        self.Y_ = np.empty((n_samples, self.n_annotators()))
        self.C_ = np.empty((n_samples, self.n_annotators()))

        # check shape and values of cluster_labelling_acc parameter
        if isinstance(cluster_labelling_acc, str):
            acc_arr = np.empty((self.n_annotators(), n_clusters, 2))
            cluster_indices = np.arange(0, n_clusters)
            if cluster_labelling_acc == 'one_hot':
                for a_idx in range(self.n_annotators()):
                    cluster_j = a_idx % n_clusters
                    acc_arr[a_idx, cluster_j] = [0.8, 1.0]
                    acc_arr[a_idx, cluster_indices != cluster_j] = [1. / n_classes, 1. / n_classes + 0.1]
            elif cluster_labelling_acc == 'equidistant':
                min_label_acc = 1. / n_classes
                label_acc_step = (1 - min_label_acc) / (self.n_annotators() + 1)
                mean_label_acc = np.linspace(min_label_acc, 1-2*label_acc_step, self.n_annotators())
                for index in np.ndindex(self.n_annotators(), n_clusters, 2):
                    if index[-1] == 0:
                        acc_arr[index] = mean_label_acc[index[0]]
                    else:
                        acc_arr[index] = mean_label_acc[index[0]] + 2 * label_acc_step
            else:
                raise ValueError(
                    'The default string options for the parameter `cluster_labelling_acc` are {one_hot, equidistant}.')
            cluster_labelling_acc = acc_arr
        self.cluster_labelling_acc_ = check_labelling_array(cluster_labelling_acc, (n_annotators, n_clusters, 2))

        # generate annotators according depending on the clustering and the corresponding labelling accuracies
        for a_idx in range(self.n_annotators_):
            for x_idx in range(np.size(self.X_, 0)):
                acc_low = self.cluster_labelling_acc_[a_idx, y_cluster_transformed[x_idx], 0]
                acc_up = self.cluster_labelling_acc_[a_idx, y_cluster_transformed[x_idx], 1]
                acc = self.random_state_.uniform(acc_low, acc_up)
                p = column_or_1d(np.array([(1 - acc) / (n_classes - 1)] * n_classes))
                p[y_true_transformed[x_idx]] = acc
                self.Y_[x_idx, a_idx] = le_cla.inverse_transform([self.random_state_.choice(range(n_classes), p=p)])
                self.C_[x_idx, a_idx] = acc

        self._add_confidence_noise(probabilistic=True)

    def labelling_performance_per_cluster(self, perf_func=None):
        """
        Computes the labelling performance per cluster of the annotators.

        Parameters
        ----------
        perf_func : callable(y_true, y_pred)
            Function evaluating the performance depending on true and predicted class labels.
            The default function is the accuracy score.

        Returns
        -------
        perfs: list, shape (n_annotators, n_clusters)
            An entry list[a_idx, c_idx] represents the actual performance of annotator a_idx on cluster c_idx.
        """
        clusters = np.unique(self.y_cluster_)
        perf_func = accuracy_score if perf_func is None else perf_func
        return [[perf_func(self.y_true_[self.y_cluster_ == clu], self.Y_[self.y_cluster_ == clu, a_idx])
                 for clu in clusters]
                for a_idx in range(self.n_annotators())]
