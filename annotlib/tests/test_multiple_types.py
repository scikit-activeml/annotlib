import unittest
import numpy as np

from annotlib.cluster_based import ClusterBasedAnnot
from annotlib.difficulty_based import DifficultyBasedAnnot
from annotlib.multi_types import MultiAnnotTypes

from sklearn.datasets import load_iris


class TestMultipleAnnotTypes(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.annotator_types = [ClusterBasedAnnot(X=self.X, y_true=self.y, random_state=42, n_annotators=3),
                                DifficultyBasedAnnot(X=self.X, y_true=self.y, random_state=42,
                                                     n_annotators=2)]
        self.y_unique = np.unique(self.y)

    def test_init(self):
        # false parameters
        self.assertRaises(TypeError, MultiAnnotTypes, annotator_types=None)
        self.assertRaises(TypeError, MultiAnnotTypes, annotator_types=[None])

        # correct annotator types
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        self.assertEqual(self.annotator_types, annotators.annotator_types_)

    def test_add_annotators(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)

        # false parameters
        self.assertRaises(TypeError, annotators.add_annotators, annotator_types=None)
        self.assertRaises(TypeError, annotators.add_annotators, annotator_types=[None])

        # correct annotator types
        annotators.add_annotators(annotator_types=ClusterBasedAnnot(X=self.X, y_true=self.y))
        self.assertEqual(self.annotator_types, annotators.annotator_types_)

    def test_n_annotators(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        self.assertEqual(self.annotator_types[0].n_annotators() + self.annotator_types[1].n_annotators(),
                         annotators.n_annotators())

    def test_n_queries(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        annotators.class_labels(self.X, annotator_ids=[0, 3], query_value=len(self.X))
        np.testing.assert_array_equal([len(self.X), 0, 0, len(self.X), 0], annotators.n_queries())

    def test_queried_samples(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        annotators.class_labels(self.X[0:2], annotator_ids=[0])
        np.testing.assert_array_equal(self.X[0:2], annotators.queried_samples()[0])

    def test_class_labels(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        Y = annotators.class_labels(self.X[0:2], annotator_ids=[0])
        Y_true = np.full((annotators.n_annotators(), 2), np.nan)
        Y_true[0, :] = self.annotator_types[0].class_labels(self.X[0:2])[:, 0]
        np.testing.assert_array_equal(Y_true.T, Y)

        Y = annotators.class_labels(self.X[0:2])
        Y_true = np.full((annotators.n_annotators(), 2), np.nan)
        Y_true[0:3, :] = self.annotator_types[0].class_labels(self.X[0:2]).T
        Y_true[3:5, :] = self.annotator_types[1].class_labels(self.X[0:2]).T
        np.testing.assert_array_equal(Y_true.T, Y)

    def test_confidence_scores(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)
        C = annotators.confidence_scores(self.X[0:2], annotator_ids=[0])
        C_true = np.full((annotators.n_annotators(), 2), np.nan)
        C_true[0, :] = self.annotator_types[0].confidence_scores(self.X[0:2])[:, 0]
        np.testing.assert_array_equal(C_true.T, C)

        C = annotators.confidence_scores(self.X[0:2])
        C_true = np.full((annotators.n_annotators(), 2), np.nan)
        C_true[0:3, :] = self.annotator_types[0].confidence_scores(self.X[0:2]).T
        C_true[3:5, :] = self.annotator_types[1].confidence_scores(self.X[0:2]).T
        np.testing.assert_array_equal(C_true.T, C)

    def test_transform_ids(self):
        annotators = MultiAnnotTypes(annotator_types=self.annotator_types)

        transformed_ids = annotators._transform_ids(annotator_ids=[0, 1, 2, 3, 4])
        true_transformed_ids = [[0, 1, 2], [0, 1]]
        np.testing.assert_array_equal(true_transformed_ids, transformed_ids)

        transformed_ids = annotators._transform_ids(annotator_ids=[1, 4])
        true_transformed_ids = [[1], [1]]
        np.testing.assert_array_equal(true_transformed_ids, transformed_ids)


if __name__ == '__main__':
        unittest.main()
