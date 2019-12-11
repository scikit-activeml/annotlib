import unittest
import numpy as np
import copy

from annotlib.cluster_based import ClusterBasedAnnot
from annotlib.dynamic import DynamicAnnot
from annotlib.multi_types import MultiAnnotTypes

from sklearn.datasets import load_iris


class TestDynamicAnnot(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.annotator = ClusterBasedAnnot(X=self.X, y_true=self.y, random_state=42)
        annotator_types = [ClusterBasedAnnot(X=self.X, y_true=self.y, random_state=42),
                           ClusterBasedAnnot(X=self.X, y_true=self.y, random_state=42)]
        self.annotator_types = MultiAnnotTypes(annotator_types=annotator_types)
        self.y_unique = np.unique(self.y)

    def test_init(self):
        # false parameters
        self.assertRaises(TypeError, DynamicAnnot, annotator_model=None, y_unique=self.y_unique)
        self.assertRaises(TypeError, DynamicAnnot, annotator_model=self.annotator, y_unique=self.y_unique,
                          adversarial=None)
        self.assertRaises(ValueError, DynamicAnnot, annotator_model=self.annotator, y_unique=self.y_unique,
                          adversarial=[False, True])
        self.assertRaises(ValueError, DynamicAnnot, annotator_model=self.annotator, y_unique=self.y_unique,
                          learning_rates=[-2])

        # default learning rates
        learning_rates_1 = DynamicAnnot(annotator_model=self.annotator, y_unique=self.y_unique,
                                        random_state=42).learning_rates_
        learning_rates_2 = DynamicAnnot(annotator_model=self.annotator, y_unique=self.y_unique,
                                        random_state=42).learning_rates_
        np.testing.assert_array_equal(learning_rates_1, learning_rates_2)
        self.assertTrue(self.annotator.n_annotators(), len(learning_rates_1))

        # test manual learning rates
        learning_rates = [.2, -.1, .3]
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator, y_unique=self.y_unique,
                                         learning_rates=learning_rates)
        np.testing.assert_array_equal(learning_rates, dynamic_annotator.learning_rates_)
        learning_rates = [.2, -.1, .3, .5, .3, .0]
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator_types, y_unique=self.y_unique,
                                         learning_rates=learning_rates)
        np.testing.assert_array_equal(learning_rates, dynamic_annotator.learning_rates_)

    def test_n_annotators(self):
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator, y_unique=self.y_unique)
        self.assertEqual(self.annotator.n_annotators(), dynamic_annotator.n_annotators())
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator_types, y_unique=self.y_unique)
        self.assertEqual(self.annotator_types.n_annotators(), dynamic_annotator.n_annotators())

    def test_n_queries(self):
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator, y_unique=self.y_unique)
        dynamic_annotator.class_labels(self.X[0:2], y_true=self.y[0:2], query_value=2)
        np.testing.assert_array_equal(self.annotator.n_queries(), dynamic_annotator.n_queries())
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator_types, y_unique=self.y_unique)
        dynamic_annotator.class_labels(self.X[0:2], y_true=self.y[0:2], query_value=2)
        np.testing.assert_array_equal(self.annotator_types.n_queries(), dynamic_annotator.n_queries())

    def test_queried_samples(self):
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator_types, y_unique=self.y_unique)
        dynamic_annotator.class_labels(self.X[0:2], y_true=self.y[0:2], query_value=2, annotator_ids=[0])
        np.testing.assert_array_equal(self.X[0:2], dynamic_annotator.queried_samples()[0])

    def test_class_labels(self):
        dynamic_annotator = DynamicAnnot(annotator_model=copy.deepcopy(self.annotator_types),
                                         y_unique=self.y_unique, learning_rates=[0., 0., -1., -1., 1., 1.],
                                         adversarial=[False, False, True, False, True, False])
        dynamic_annotator.class_labels(self.X, y_true=self.y, query_value=len(self.X), annotator_ids=[0, 2, 3, 4])
        labelling_accuracies = np.array(dynamic_annotator.labelling_performance(X=self.X, y_true=self.y))
        np.testing.assert_array_equal([0., 1.], labelling_accuracies[[2, 4]])
        self.assertGreater(labelling_accuracies[3], 0)

    def test_confidence_scores(self):
        dynamic_annotator = DynamicAnnot(annotator_model=self.annotator_types, y_unique=self.y_unique)
        np.testing.assert_array_equal(self.annotator_types.confidence_scores(self.X),
                                      dynamic_annotator.confidence_scores(self.X))


if __name__ == '__main__':
        unittest.main()
