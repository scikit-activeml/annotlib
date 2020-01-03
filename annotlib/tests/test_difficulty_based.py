import unittest
import numpy as np

from annotlib.difficulty_based import DifficultyBasedAnnot

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier


class TestDifficultyBasedAnnot(unittest.TestCase):

    def test_init(self):
        X, y = load_iris(return_X_y=True)

        # false parameters
        self.assertRaises(ValueError, DifficultyBasedAnnot, X, y, n_annotators=-1)
        self.assertRaises(ValueError, DifficultyBasedAnnot, X, y, n_annotators=2, alphas=[0.2])
        self.assertRaises(TypeError, DifficultyBasedAnnot, X, y, classifiers=KMeans)

        # default alpha parameters
        annotator = DifficultyBasedAnnot(X=X, y_true=y, n_annotators=3)
        self.assertEqual(3, len(annotator.alphas_))
        self.assertEqual(len(annotator.alphas_), annotator.n_annotators())
        self.assertTrue((annotator.betas_ >= 0).all())
        self.assertEqual(len(X), len(annotator.betas_))
        np.testing.assert_array_equal([0, 1, 2], annotator.alphas_)
        self.assertTrue((annotator.C_ >= 0).all() and (annotator.C_ <= 1).all)

        # estimating simplicity scores by means of multiple classifiers
        annotator = DifficultyBasedAnnot(X=X, y_true=y, classifiers=[SVC(probability=True, gamma='auto'),
                                                                     GaussianProcessClassifier()])
        labelling_acc = annotator.labelling_performance(X=X, y_true=y)
        self.assertEqual(5, len(annotator.alphas_))
        self.assertEqual(len(annotator.alphas_), annotator.n_annotators())
        self.assertTrue((annotator.betas_ >= 0).all())
        self.assertEqual(len(X), len(annotator.betas_))
        self.assertTrue((annotator.C_ >= 0).all() and (annotator.C_ <= 1).all)
        self.assertGreater(labelling_acc[-1], labelling_acc[0])

    def test_plot_annotators_labelling_probabilities(self):
        X, y = load_iris(return_X_y=True)
        annotator = DifficultyBasedAnnot(X=X, y_true=y)
        fig, ax = annotator.plot_annotators_labelling_probabilities()
        self.assertEqual(3, fig.get_figheight())
        self.assertEqual(5, fig.get_figwidth())


if __name__ == '__main__':
    unittest.main()
