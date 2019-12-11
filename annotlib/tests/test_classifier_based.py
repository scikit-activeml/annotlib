import unittest
import numpy as np

from annotlib.classifier_based import ClassifierBasedAnnot
from sklearn.datasets import load_iris
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.svm import SVC


class TestClassifierBasedAnnot(unittest.TestCase):

    def test_init(self):
        X, y = load_iris(return_X_y=True)

        # false parameters
        self.assertRaises(TypeError, ClassifierBasedAnnot, X, y, classifiers=KMeans)
        self.assertRaises(ValueError, ClassifierBasedAnnot, X, y, n_annotators=-1)
        self.assertRaises(ValueError, ClassifierBasedAnnot, X, y, n_annotators=2,
                          train_ratios=[[.4, .6, .5], [.8, 2, .5]])
        self.assertRaises(ValueError, ClassifierBasedAnnot, X, y, n_annotators=3,
                          train_ratios=[[.4, .6], [.8, .4]])
        self.assertRaises(TypeError, ClassifierBasedAnnot, X, y, n_annotators=1, features=[[0, -1, -1, -1]])
        self.assertRaises(TypeError, ClassifierBasedAnnot, X, y,
                          classifiers=[SVC(probability=True), SVC(probability=True)])

        # default train ratio parameters
        annotator = ClassifierBasedAnnot(X=X, y_true=y, n_annotators=2, train_ratios='one-hot')
        true_train_ratios = np.array([[1, .2, .2], [.2, 1, .2]])
        np.testing.assert_array_equal(true_train_ratios, annotator.train_ratios_)
        annotator = ClassifierBasedAnnot(X=X, y_true=y, n_annotators=3, train_ratios='equidistant')
        true_train_ratios = [[.333, .333, .333], [.666, .666, .666], [1, 1, 1]]
        np.testing.assert_allclose(true_train_ratios, annotator.train_ratios_, rtol=1.e-02)

        # default classification model
        annotator = ClassifierBasedAnnot(X=X, y_true=y, n_annotators=3)
        self.assertEqual(annotator.n_annotators(), len(annotator.classifiers_))
        self.assertTrue((annotator.C_ >= 0).all() and (annotator.C_ <= 1).all)
        self.assertEqual(3, len(annotator.classifiers_))
        self.assertNotEqual(annotator.classifiers_[0], annotator.classifiers_[1])

        # test manual train ratios and features
        train_ratios = [[.2, .0, .4], [.6, .9, 1]]
        features = np.full((2, np.size(X, 1)), True)
        features[0, 0] = False
        annotator = ClassifierBasedAnnot(X=X, y_true=y, n_annotators=2, train_ratios=train_ratios,
                                         features=features)
        np.testing.assert_equal(train_ratios, annotator.train_ratios_)
        self.assertTrue(check_is_fitted(annotator.classifiers_[0], 'support_') is None)
        self.assertTrue(check_is_fitted(annotator.classifiers_[1], 'support_') is None)
        self.assertEqual(np.size(X, 1)-1, np.size(annotator.classifiers_[0].support_vectors_, 1))
        self.assertEqual(np.size(X, 1), np.size(annotator.classifiers_[1].support_vectors_, 1))


if __name__ == '__main__':
    unittest.main()
