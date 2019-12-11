import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from abc import ABC, abstractmethod

from annotlib.utils import check_indices

from sklearn.utils import check_array, check_X_y
from sklearn.metrics import accuracy_score, confusion_matrix


class BaseAnnot(ABC):
    """BaseAnnot

    This class is an abstract representation of base annotators.
    It defines the required methods and implements methods to evaluate the labelling performance and to illustrate the
    samples labelled by the annotators.
    """

    @abstractmethod
    def n_annotators(self):
        """Method returning the number of annotators.

        Returns
        -------
        n_annotators: int
            Number of annotators.
        """
        pass

    @abstractmethod
    def n_queries(self):
        """Method returning the number of queries posed to an annotator.

        Returns
        -------
        n_queries: numpy.ndarray, shape (n_annotators)
            An entry `n_queries[a_idx]` indicates how many queries annotator with id `a_idx` has processed.
        """
        pass

    @abstractmethod
    def queried_samples(self):
        """Method returning the samples for which the annotators were queried to provide class labels.

        Returns
        -------
        X_queried: list, shape (n_annotators, n_queried_samples, n_features)
            An entry `X_queried[a_idx]` represents the samples for which the annotator with id `a_idx` was queried to
            provide class labels.
        """
        pass

    @abstractmethod
    def class_labels(self, X, annotator_ids, query_value, **kwargs):
        """Method returning the class labels of the given samples.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples whose class labels are queried.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose class labels are queried.
        query_value: int
            The `query_value` represents the increment of the query statistics of the queried annotators.

        Returns
        -------
        Y: numpy.ndarray, shape (n_samples, n_annotators)
            Class labels of the given samples which were provided by the queried annotators.
            The non queried annotators should return `np.nan` values.
        """
        pass

    @abstractmethod
    def confidence_scores(self, X, annotator_ids, **kwargs):
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
            The non queried annotators should return `np.nan` values.
        """
        pass

    def labelling_performance(self, X, y_true, perf_func=None, **kwargs):
        """Method computing the labelling performance of each annotator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples on which the labelling performances of the annotators is evaluated.
        y_true: array-like, shape (n_samples)
            True class labels of the given samples.
        perf_func: callable(y_true, y_pred)
            Function evaluating the performance depending on true and predicted class labels.
            The default function is the accuracy score.

        Returns
        -------
        perf: list, shape (n_annotators)
            Label accuracy of each annotator.
        """
        perf_func = accuracy_score if perf_func is None else perf_func
        X, y_true = check_X_y(X, y_true)
        Y = self.class_labels(X, query_value=0, y_true=y_true, **kwargs)
        perf = []
        for a in range(self.n_annotators()):
            not_nan = ~np.isnan(Y[:, a])
            if not_nan.any():
                perf.append(perf_func(y_true=y_true[not_nan], y_pred=Y[not_nan, a]))
            else:
                perf.append(np.nan)
        return perf

    def plot_labelling_accuracy(self, X, y_true, annotator_ids=None, figsize=(4, 4), dpi=150, fontsize=12):
        """Method plotting the labelling accuracy of each desired annotator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples on which the labelling accuracies of the annotators is evaluated.
        y_true: array-like, shape (n_samples)
            True class labels of the given samples.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose labelling accuracies are plotted.
        figsize: 2-tuple of floats, default: (4, 4
            Figure dimension (width, height) in inches.
        dpi: float, default: 150
            Dots per inch.
        fontsize: int
            Font size of plotted text.

        Returns
        -------
        fig: matplotlib.figure.Figure object
            Created figure.

        ax: :py:class:`matplotlib.axes.Axes` object
            Created axes.
        """
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')
        acc = np.asarray(self.labelling_performance(X, y_true))
        acc[np.isnan(acc)] = 0
        x = np.arange(self.n_annotators())
        annot_names = [r'annotator $a_' + str(a_idx) + '$' for a_idx in annotator_ids]
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.bar(x, acc[annotator_ids])
        plt.xticks(x, annot_names, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('labelling accuracy', fontsize=fontsize)
        plt.title('labelling accuracy of annotators', fontsize=fontsize)
        return fig, ax

    def plot_labelling_confusion_matrices(self, X, y_true, y_unique, annotator_ids=None, figsize=(4, 4), dpi=150,
                                          fontsize=12):
        """Method plotting the labelling confusion matrix of each desired annotator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
             Samples on which the labelling confusion matrices of the annotators is evaluated.
        y_true: array-like, shape (n_samples)
            True class labels of the given samples.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose labelling confusion matrices are plotted.
        figsize: 2-tuple of floats, default: (4, 4*n_annotators)
            Figure dimension (width, height*n_annotators) in inches.
        dpi: float, default: 150
            Dots per inch.
        fontsize: int
            Font size of plotted text.

        Returns
        -------
        fig: matplotlib.figure.Figure object
            Created figure.

        ax: array-like, shape (n_annotator_ids).
            The array ax is a collection of :py:class:`matplotlib.axes.Axes` instances representing the plots of
            the annotators.
        """
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')
        conf_matrices = self.labelling_performance(X, y_true, perf_func=confusion_matrix)
        n_classes = len(y_unique)
        for matrix_idx in range(len(conf_matrices)):
            if not isinstance(conf_matrices[matrix_idx], np.ndarray):
                conf_matrices[matrix_idx] = np.zeros((n_classes, n_classes), dtype=int)
        y_unique = np.sort(y_unique)
        fig, ax = plt.subplots(len(annotator_ids), 1, figsize=(figsize[0], figsize[1]*len(annotator_ids)),
                               dpi=dpi)
        ax = [ax] if len(annotator_ids) == 1 else ax
        for i, a_idx in enumerate(annotator_ids):
            df_cm = pd.DataFrame(conf_matrices[a_idx], range(n_classes), range(n_classes))
            sn.heatmap(df_cm, annot=True, annot_kws={"size": fontsize}, xticklabels=y_unique, yticklabels=y_unique,
                       cmap="YlGnBu", fmt="d", ax=ax[i], cbar=False)
            ax[i].tick_params(labelsize=fontsize)
            ax[i].set_xlabel('predicted class labels', fontsize=fontsize)
            ax[i].set_ylabel('true class labels', fontsize=fontsize)
            ax[i].set_title('confusion matrix of annotator $a_' + str(a_idx) + '$', fontsize=fontsize)
        return fig, ax

    def plot_class_labels(self, X, features_ids=None, annotator_ids=None, plot_confidences=5, y_true=None,
                          figsize=(5, 3), dpi=150, fontsize=7, **kwargs):
        """Method creating scatter plots of the given samples for each annotator.
        In each scatter plot, the samples are colored according the class labels provided by the
        corresponding annotator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Samples which are plotted.
        features_ids: array-like, shape (2)
            The feature indices to be plotted. The array is limited to two indices.
        annotator_ids: array-like, shape (n_queried_annotators)
            The indices of the annotators whose class label distributions are plotted.
        plot_confidences: boolean
            If true, the size of the markers is plotted according to the given confidence scores.
        y_true: array-like, shape (n_samples)
            This is a optional parameter. If the true class labels are given, the samples are marked according to
            correctness of their predicted class label
        figsize: 2-tuple of floats, default: (5, 3*n_annotators)
            Figure dimension (width, height*n_annotators) in inches.
        dpi: float, default: 150
            Dots per inch.
        fontsize: int
            Font size of plotted text.

        Returns
        -------
        fig: matplotlib.axes.Axes object
            Created figure.

        ax: array-like, shape (n_annotator_ids).
            The array ax is a collection of :py:class:`matplotlib.axes.Axes` instances representing the plots of
            the annotators.
        """
        # check annotator_ids
        annotator_ids = check_indices(annotator_ids, self.n_annotators() - 1, 'annotator_ids')

        # check features_ids
        X = check_array(X)
        features_ids = np.arange(1 + (np.size(X, 1) > 1), dtype=int) if features_ids is None else features_ids
        n_features = len(features_ids)
        features_ids = check_indices(features_ids, n_features - 1, 'feature_ids')
        x_0 = X[:, features_ids[0]]
        if n_features > 1:
            x_1 = X[:, features_ids[1]]
        else:
            x_1 = np.zeros(np.size(X, 0))

        # check true class labels
        # y_true = kwargs.get('y_true', None)
        if y_true is not None:
            X, y_true = check_X_y(X, y_true)
        m_true = 'o'
        m_false = 'x'

        # query class labels
        Y = self.class_labels(X=X, annotator_ids=annotator_ids, query_value=0, y_true=y_true, **kwargs)
        C = self.confidence_scores(X=X, annotator_ids=annotator_ids)
        y_unique = np.unique(Y[~np.isnan(Y)])
        n_classes = len(y_unique)

        # setup the scatter plots
        fig, ax = plt.subplots(len(annotator_ids), 1, figsize=(figsize[0], len(annotator_ids) * figsize[1]), dpi=dpi)
        ax = [ax] if len(annotator_ids) == 1 else ax
        colors = cm.rainbow(np.linspace(0, 1, n_classes + 1))
        for a in range(len(annotator_ids)):
            y_a = Y[:, annotator_ids[a]]
            if plot_confidences:
                c_a = C[:, annotator_ids[a]]
                c_a = (.015*dpi + .0175*dpi * c_a / np.nanmax(C)) ** (350/dpi)
                c_a[np.isnan(c_a)] = 1
            else:
                c_a = np.full(len(y_a), .1*dpi)
            if y_true is None:
                for i, y in enumerate(y_unique):
                    flag = y_a == y
                    ax[a].scatter(x_0[flag], x_1[flag], color=colors[i].reshape(1, -1),
                                  label='prediction is' + str(int(y)), s=c_a[flag])
            else:
                for i, y in enumerate(y_unique):
                    true_predictions = np.logical_and(y_a == y, y_a == y_true)
                    false_predictions = np.logical_and(y_a == y, y_a != y_true)
                    ax[a].scatter(x_0[true_predictions], x_1[true_predictions], color=colors[i].reshape(1, -1),
                                  label='prediction ' + str(int(y)) + ' is true', marker=m_true,
                                  s=c_a[true_predictions])
                    ax[a].scatter(x_0[false_predictions], x_1[false_predictions], color=colors[i].reshape(1, -1),
                                  label='prediction ' + str(int(y)) + ' is false', marker=m_false,
                                  s=c_a[false_predictions])
            if np.sum(np.isnan(y_a)) > 0:
                ax[a].scatter(x_0[np.isnan(y_a)], x_1[np.isnan(y_a)], color=colors[n_classes].reshape(1, -1),
                              label='prediction is NA', marker=m_false, s=np.full(np.sum(np.isnan(y_a)), 5))
            lgnd = ax[a].legend(loc='best', fancybox=False, framealpha=0.5, prop={'size': fontsize})
            for handle in lgnd.legendHandles:
                handle.set_sizes([fontsize])
            ax[a].tick_params(labelsize=fontsize)
            ax[a].set_title(r'class labels predicted by annotator $a_' + str(annotator_ids[a]) + '$', fontsize=fontsize)

        return fig, ax