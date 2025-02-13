def array(object, dtype=None):
    shape = shape if (shape := calc_shape_from_obj(object)) else (1,)
    array_like = []

    def _flatten(data):
        if isinstance(data, (tuple, list)):
            for item in data:
                _flatten(item)
        else:
            array_like.append(data)

    _flatten(object)
    if dtype is None:
        dtype = (
            int32
            if all(isinstance(idx, int) for idx in array_like)
            else float32
        )
    out = ndarray(shape, dtype)
    out[:] = array_like
    return out
