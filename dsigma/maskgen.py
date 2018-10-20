"""Mask for runComputeDS."""
import operator

import numpy as np

__all__ = ["MaskError", "compute_mask"]


class MaskError(Exception):
    """Error class for compute mask."""

    pass


def compute_mask(lenses, mask_args):
    """Object mask for runComputeDS.

    Parameters
    ----------
    lenses : numpy array
        Lens catalog.
    mask_args : string
        String that described the lens selection.

    Return
    ------
    mask : boolen array
        Mask array
    """
    mask = np.ones(len(lenses), dtype=bool)
    if mask_args == "":
        return mask

    # select lenses where "field/operation/value:field/operation/value"
    # Value is assumed to be a float
    # supported operations are >, <, ==
    op_splitter = ":"   # splits operations that will be anded together
    arg_splitter = "/"  # splits operation into args
    ops = mask_args.split(op_splitter)
    ops = [i.split(arg_splitter) for i in ops]

    op_lookup = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        }

    for op in ops:
        if len(op) != 3:
            raise MaskError("Misformed op '{}'".format(op))
        try:
            field, operation, value = op[0], op[1], float(op[2])

        except ValueError:
            raise MaskError(
                "value must be convertable to float, got '{}'".format(op[2])
                )

        try:
            op_mask = op_lookup[operation](lenses[field], value)
        except KeyError:
            raise MaskError("Unsupported operation '{}'".format(operation))

        mask = (mask & op_mask)

    return mask
