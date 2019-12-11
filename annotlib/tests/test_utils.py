import unittest
import numpy as np

from annotlib.utils import check_range, check_positive_integer, check_indices, check_shape, check_labelling_array, \
    transform_confidences


class TestUtils(unittest.TestCase):

    def test_check_range(self):
        arr = [0.1, 0.5, 1.0, 0.9, 0.0]
        self.assertRaises(ValueError, check_range, arr, 0, 0.9)
        np.testing.assert_array_equal(arr, check_range(arr, 0, 1))

    def test_check_indices(self):
        indices = check_indices(None, max_index=4)
        np.testing.assert_array_equal(list(range(5)), indices)

        indices = check_indices([2, 3], max_index=4)
        np.testing.assert_array_equal([2, 3], indices)

        indices = [3, 3]
        self.assertRaises(ValueError, check_indices, indices=indices, max_index=5)

        indices = [2, 6]
        self.assertRaises(ValueError, check_indices, indices=indices, max_index=5)

    def test_check_positive_integer(self):
        self.assertRaises(ValueError, check_positive_integer, value=-4)
        self.assertRaises(ValueError, check_positive_integer, value=2.3)
        self.assertEqual(2, check_positive_integer(value=2))

    def test_check_shape(self):
        arr = [[2, 2], [3, 4]]
        self.assertRaises(ValueError, check_shape, arr=arr, shape=[2, 3])
        np.testing.assert_array_equal(arr, check_shape(arr=arr, shape=[2, 2]))

    def test_check_labelling_array(self):
        arr = [[.2, .4], [.2, .2]]
        self.assertRaises(ValueError, check_labelling_array, arr=arr, shape=[3, 1])
        arr = [[2, .3], [2, 2]]
        self.assertRaises(ValueError, check_labelling_array, arr=arr, shape=[2, 2])
        arr = [[.0, .3], [1., .2]]
        np.testing.assert_array_equal(arr, check_labelling_array(arr, shape=[2, 2]))

    def test_transform_confidences(self):
        C = [[0., 1.], [.4, .8]]
        C_trans = [[1., 1.], [.2, .6]]
        C_result = transform_confidences(C, 2)
        np.testing.assert_almost_equal(C_trans, C_result)


if __name__ == '__main__':
    unittest.main()
