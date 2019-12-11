import unittest
import numpy as np

from annotlib.cluster_based import ClusterBasedAnnot
from sklearn.datasets import load_iris


class TestClusterBasedAnnot(unittest.TestCase):

    def test_init(self):
        X, y = load_iris(return_X_y=True)

        # test wrong parameters
        self.assertRaises(ValueError, ClusterBasedAnnot, X, y, n_annotators=-1)
        self.assertRaises(ValueError, ClusterBasedAnnot, X, y, cluster_labelling_acc=[[.4, .6]])
        self.assertRaises(ValueError, ClusterBasedAnnot, X, y, cluster_labelling_acc='a')

        # test default labelling accuracies
        annotator = ClusterBasedAnnot(X=X, y_true=y, cluster_labelling_acc='one_hot', n_annotators=1)
        cluster_labelling_acc = [[[.8, 1.],
                                  [.333, .433],
                                  [.333, .433]]]
        np.testing.assert_allclose(cluster_labelling_acc, annotator.cluster_labelling_acc_, rtol=1.e-2)
        self.assertTrue(np.logical_and(annotator.C_ >= 0, annotator.C_ <= 1).all())
        annotator = ClusterBasedAnnot(X=X, y_true=y, cluster_labelling_acc='equidistant', n_annotators=1)
        cluster_labelling_acc = [[[.333, 1.],
                                  [0.333, 1.],
                                  [0.333, 1.]]]
        self.assertTrue(np.logical_and(annotator.C_ >= 0, annotator.C_ <= 1).all())
        np.testing.assert_allclose(cluster_labelling_acc, annotator.cluster_labelling_acc_, rtol=1.e-2)

        # test manual defined clustering labelling accuracies
        cluster_labelling_acc = [[[0., 0.]]]
        y_cluster = np.ones(len(X))
        annotator = ClusterBasedAnnot(X=X, y_true=y, y_cluster=y_cluster,
                                      cluster_labelling_acc=cluster_labelling_acc, n_annotators=1)
        np.testing.assert_array_equal(cluster_labelling_acc, annotator.cluster_labelling_acc_)
        self.assertEqual(0., annotator.labelling_performance(X=X, y_true=y)[0])

    def test_labelling_performance_per_cluster(self):
        X, y = load_iris(return_X_y=True)
        cluster_labelling_acc = [[[0., 0.],
                                  [0., 0.],
                                  [1., 1.]],
                                 [[1., 1.],
                                  [1., 1.],
                                  [0., 0.]]]
        annotator = ClusterBasedAnnot(X=X, y_true=y, n_annotators=2,
                                      cluster_labelling_acc=cluster_labelling_acc)
        labelling_perf_per_cluster = annotator.labelling_performance_per_cluster()
        true_labelling_perf_per_cluster = [[0., 0., 1.],
                                           [1., 1., 0.]]
        np.testing.assert_array_equal(true_labelling_perf_per_cluster, labelling_perf_per_cluster)


if __name__ == '__main__':
    unittest.main()
