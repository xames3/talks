# utils.py

import math


def calc_size(size):
	return math.prod(size)


def calc_strides(size, itemsize):
	strides = []
	stride = itemsize
	for dim in reversed(size):
		strides.append(stride)
		stride *= dim
	return tuple(reversed(strides))


def get_step_size(view):
    contiguous = calc_strides(view._shape, view._itemsize)
    step = view._strides[-1] // contiguous[-1]
    strides = tuple(stride * step for stride in contiguous)
    return step if view._strides == strides else 0


def calc_shape_from_obj(object):
    shape = []

    def _calc_shape(e, axis):
        if isinstance(e, (tuple, list)) and not isinstance(e, (str, bytes)):
            if len(shape) <= axis:
                shape.append(0)
            current = len(e)
            if current > shape[axis]:
                shape[axis] = current
            for idx in e:
                _calc_shape(idx, axis + 1)

    _calc_shape(object, 0)
    return tuple(shape)


def broadcast_shape(input, other):
    buffer = []
    r_input = list(reversed(input))
    r_other = list(reversed(other))
    maximum = max(len(r_input), len(r_other))
    r_input.extend([1] * (maximum - len(r_input)))
    r_other.extend([1] * (maximum - len(r_other)))
    for idx, jdx in zip(r_input, r_other):
        if idx == jdx or idx == 1 or jdx == 1:
            buffer.append(max(idx, jdx))
        else:
            raise ValueError(
                f"Operands couldn't broadcast together with shapes {input} "
                f"and {other}"
            )
    return tuple(reversed(buffer))
