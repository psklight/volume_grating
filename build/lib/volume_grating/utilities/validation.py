import numpy as np


def validate_input_numeric(value, shape=None):
    """
    Validate whether the input value is a list of numbers of a ndarry of numbers, against a requires shape (if ``shape`` is not ``None``).
    When ``shape`` is a tuple, for example (4, None, 2), it checks ``value``'s shape that ``value`` should have 3 dimensions.
    The first dimension must have a length of 4, the second dimension can have any positive length, and the thir
    dimension must have a length of 2.

    :param value: an input to validate
    :param shape: [default = None]. If not ``None``, must be a tuple.
    :return is_valid: True if ``value`` is a list/ndarray of numeric values with the right shape.
    :return failmessage: a failure message of the validation.
    """
    isvalid = True
    failmessage = ""

    if isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            value = np.array(value, dtype=np.float)
        else:
            isvalid = False
            failmessage = "Input is not numeric."
    assert isinstance(shape, tuple) or shape is None, 'shape must be None or a tuple of integers and/or None.'

    if isinstance(value, np.ndarray):
        try:
            value = value.astype(np.float)
            if shape is not None:
                inputshape = value.shape
                try:
                    for i in range(len(shape)):
                        if shape[i] is not None and shape[i]!=inputshape[i]:
                            isvalid = False
                            failmessage = "Input (shape of {}) does not match a shape of {}.".format(inputshape, shape)
                            break
                except:
                    isvalid = False
                    failmessage = "Input (shape of {}) does not match the shape of {}.".format(inputshape, shape)
        except ValueError:
            isvalid = False
            failmessage = "Input is not numeric."
    else:
        isvalid = False
        failmessage = "Input is not numeric."

    is_valid = isvalid

    return is_valid, failmessage


