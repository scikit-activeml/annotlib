import unittest
import numpy as np

from annotlib.standard import StandardAnnot


class TestStandardAnnot(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(10).reshape(5, 2)
        self.y_true = np.asarray([0, 1, 2, 0, 1])
        self.Y = np.asarray([[0, 1, 2, 0, 1], [1, 2, 2, 1, 0]]).T
        self.C = np.asarray([[0.25, 0.7, 0.6, 0.8, 0.9], [0.6, 0.3, 0.9, 1, 0.4]]).T

    def test_init(self):
        # test initialisation with false parameters
        self.assertRaises(ValueError, StandardAnnot, self.X, self.Y[:3], self.C[:3])
        self.assertRaises(ValueError, StandardAnnot, self.X, self.Y, self.C[:3])
        self.assertRaises(ValueError, StandardAnnot, self.X, self.Y[:, 0].reshape(-1, 1), self.C)

        # test initialisation with correct parameters
        self.assertEqual(StandardAnnot(self.X, self.Y, self.C).n_annotators(), 2)
        np.testing.assert_array_equal(self.C.shape, StandardAnnot(self.X, self.Y).C_.shape)

    def test_class_labels(self):
        annotator = StandardAnnot(self.X, self.Y, self.C)

        # test querying class labels
        ids = [0, 2, 3]
        X = self.X[ids]
        Y = annotator.class_labels(X)
        np.testing.assert_array_equal(self.Y[ids], Y)

        # test querying class labels of missing samples
        X = np.array([[-1, -1], [-2, -3]])
        Y = annotator.class_labels(X)
        np.testing.assert_array_equal(np.array([[np.nan, np.nan], [np.nan, np.nan]]), Y)

        # test querying class labels of selected annotators
        ids = [0]
        Y = annotator.class_labels(self.X[0:2], ids)
        np.testing.assert_array_equal(np.array([[self.Y[0, 0], np.nan], [self.Y[0, 1], np.nan]]), Y)

    def test_confidence_scores(self):
        annotator = StandardAnnot(self.X, self.Y, self.C)

        # test querying confidence scores
        ids = [0, 2, 3]
        X = self.X[ids]
        C = annotator.confidence_scores(X)
        np.testing.assert_array_equal(self.C[ids], C)

        # test querying class labels of missing samples
        X = np.array([[-1, -1], [-2, -3]])
        C = annotator.confidence_scores(X)
        np.testing.assert_array_equal(np.array([[np.nan, np.nan], [np.nan, np.nan]]), C)

        # test querying class labels of selected annotators
        ids = [0]
        C = annotator.confidence_scores(self.X[0:2], ids)
        np.testing.assert_array_equal(np.array([[self.C[0, 0], np.nan], [self.C[1, 0], np.nan]]), C)

    def test_queried_samples(self):
        annotator = StandardAnnot(self.X, self.Y, self.C)

        # test querying class labels of selected annotators
        ids = [0]
        annotator.class_labels(self.X[0:2], ids)

        # test queried samples
        np.testing.assert_array_equal(self.X[0:2], annotator.queried_samples()[0])
        np.testing.assert_array_equal(np.array([]).reshape(0, 2), annotator.queried_samples()[1])

    def test_n_queries(self):
        annotator = StandardAnnot(self.X, self.Y, self.C)

        # test querying class labels of selected annotators
        ids = [0]
        annotator.class_labels(self.X[0:2], ids, query_value=3)

        # test number of queries
        np.testing.assert_array_equal([3, 0], annotator.n_queries())

    def test_confidence_noise(self):
        # test wrong confidences
        self.assertRaises(ValueError, StandardAnnot, self.X, self.Y, self.C, [.2, .3, .5], 42, False)

        # test correct confidences
        annotator = StandardAnnot(self.X, self.Y, np.copy(self.C), [.3, 200], 42, True)
        self.assertTrue(np.logical_and(annotator.C_ >= 0, annotator.C_ <= 1).all())


if __name__ == '__main__':
    unittest.main()

