"""
This module is a collection of helpful functions.
"""
import numpy as np

from sklearn.utils import column_or_1d


def check_range(arr, min_value, max_value, parameter_name='arr'):
    """This function tests whether all elements of an array are in a given range.

    Parameters
    ----------
    arr: array-like,
        Array.
    min_value: float
        Minimal number.
    max_value: float
        Maximal number.
    parameter_name: str,
        The name of the array arr, which is printed in case of error.

    Returns
    -------
    arr: numpy.ndarray
        Is returned, if min <= arr[i] <= max for all arr[i].
    """
    interval = [min_value, max_value]
    arr = np.asarray(arr)
    if (arr < min_value).any() or (arr > max_value).any():
        raise ValueError(
            'The parameter `' + str(parameter_name) + '` must contain values of the interval ' + str(interval) + '.')
    return arr


def check_indices(indices, max_index, parameter_name='indices'):
    """This function checks whether the given indices are valid.

    Parameters
    ----------
    indices: array-like, shape (n_indices)
        Array of indices to test.
    max_index: int
        The maximal allowed index.
    parameter_name: str,
        The name of the indices array, which is printed in case of error.

    Returns
    -------
    indices: numpy.ndarray, shape (n_indices)
        Is returned, if 0<= indices[i] <= max_index for all i and if indices contains no duplicates.
    """
    if indices is None:
        return np.arange(max_index+1, dtype=int)
    else:
        indices = column_or_1d(indices)
        indices.astype(int, casting='safe')
        indices_unique = np.unique(indices)
        if len(indices) != len(indices_unique):
            raise ValueError('The parameter `' + str(parameter_name) + '` is not allowed to have duplicates.')
        if not (indices >= 0).all() or not (indices <= max_index).all():
            raise ValueError('The parameter `' + str(parameter_name) + '` must be in the range [0, max_index].')
        return indices


def check_positive_integer(value, parameter_name='value'):
    """This function checks whether the given value is a positive integer.

    Parameters
    ----------
    value: numeric,
        Value to check.
    parameter_name: str,
        The name of the indices array, which is printed in case of error.

    Returns
    -------
    value: int,
        Checked and converted int.
    """
    int_value = int(value)
    if value < 0 or value != int_value:
        raise ValueError('The parameter `' + str(parameter_name) + '` must be a positive integer.')
    return value


def check_shape(arr, shape, parameter_name='arr'):
    """This function checks whether the given array has the expected shape.

    Parameters
    ----------
    arr: array-like
        Array whose shape is checked.
    shape: array-like
        The expected shape of the given array arr.
    parameter_name: str,
        The name of the indices array, which is printed in case of error.

    Returns
    -------
    arr: array-like
        Array whose shape is checked.
    """
    arr = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    if not np.array_equal(arr.shape, shape):
        raise ValueError('The parameter `' + str(parameter_name) + '` must have the shape '+str(shape)+'.')
    return arr


def check_labelling_array(arr, shape, parameter_name='arr'):
    """This function checks whether the given array has the given shape and all its values are in the interval [0, 1].

    Parameters
    ----------
    arr: array-like
        Array whose shape is checked.
    shape: array-like
        The expected shape of the given array arr.
    parameter_name: str,
        The name of the indices array, which is printed in case of error.

    Returns
    -------
    arr: array-like
        Array whose shape is checked.
    """
    arr = np.asarray(arr)
    arr = check_range(arr, 0, 1, parameter_name=parameter_name)
    arr = check_shape(arr, shape, parameter_name=parameter_name)
    return arr


def transform_confidences(C, n_classes):
    """Originally, non-adversarial annotators provide confidences in the interval [1/n_classes, 1].
    In contrast, adversarial annotators have confidences in [0, 1/n_classes].
    This function transforms the confidences into an interval [0, 1] for both annotator types.
    However, the meaning of the confidences is contrary. A class label provided by an adversarial annotator
    with confidence 1 is wrong in any case, whereas a class label provided by a non-adversarial annotator
    with confidence 1 is true in any case.

    Parameters
    ----------
    C: array-like, shape (n_samples, n_annotators)
        confidences to be transformed.
    n_classes: int
        Number of classes.

    Returns
    -------
    C_trans: array-like, shape (n_samples, n_annotators)
        Transformed confidences.
    """
    n_classes = check_positive_integer(n_classes, parameter_name='n_classes')
    C = np.asarray(C)
    min_proba = 1./n_classes
    C_trans = np.zeros_like(C)
    C_trans[C >= min_proba] = (C[C >= min_proba]-min_proba)/(1-min_proba)
    C_trans[C < min_proba] = (C[C < min_proba]-min_proba) / (-1 * min_proba)

    return C_trans
