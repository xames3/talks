    def _calculate_offset_and_strides(self, key):
        axis = 0
        offset = self._offset
        shape = []
        strides = []
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            ellipsis = key.index(Ellipsis)
            pre = key[:ellipsis]
            post = key[ellipsis + 1 :]
            count = len(self._shape) - len(pre) - len(post)
            if count < 0:
                raise IndexError("Too many indices for array")
            key = pre + (slice(None),) * count + post
        for dim in key:
            if axis >= len(self._shape) and dim is not None:
                raise IndexError("Too many indices for array")
            axissize = self._shape[axis] if axis < len(self._shape) else None
            if isinstance(dim, int) and axissize is not None:
                if not (-axissize <= dim < axissize):
                    raise IndexError(
                        f"Index {dim} out of bounds for axis {axis}"
                    )
                dim = dim + axissize if dim < 0 else dim
                offset += dim * self._strides[axis] // self._itemsize
                axis += 1
            elif isinstance(dim, slice) and axissize is not None:
                start, stop, step = dim.indices(axissize)
                shape.append(-(-(stop - start) // step))
                strides.append(step * self._strides[axis])
                offset += start * self._strides[axis] // self._itemsize
                axis += 1
            elif dim is None:
                shape.append(1)
                strides.append(0)
            else:
                raise TypeError(f"Invalid index type: {type(dim).__name__!r}")
        shape.extend(self._shape[axis:])
        strides.extend(self._strides[axis:])
        return offset, tuple(shape), tuple(strides)

    def __getitem__(self, key):
        offset, shape, strides = self._calculate_offset_and_strides(key)
        if not shape:
            return self._data[offset]
        return ndarray(
            shape,
            self._dtype,
            buffer=self,
            offset=offset,
            strides=strides,
        )

    def __setitem__(self, key, value):
        offset, shape, strides = self._calculate_offset_and_strides(key)
        if not shape:
            self._data[offset] = round(value, 4)
            return
        view = ndarray(
            shape,
            self._dtype,
            buffer=self,
            offset=offset,
            strides=strides,
        )
        if isinstance(value, (float, int)):
            values = [value] * calc_size(view._shape)
        elif isinstance(value, (tuple, list)):
            values = list(value)
        else:
            if not isinstance(value, ndarray):
                value = ndarray(
                    value,
                    self._dtype,
                    buffer=self,
                    offset=offset,
                    strides=strides,
                )
            values = value._flat()
        if calc_size(view._shape) != len(values):
            raise ValueError(
                "Number of elements in the value doesn't match the shape"
            )
        subviews = [view]
        idx = 0
        while subviews:
            subview = subviews.pop(0)
            if step_size := get_step_size(subview):
                block = values[idx : idx + calc_size(subview._shape)]
                converted = []
                for element in block:
                    if not self._dtype.numpy.startswith(("float", "bool")):
                        converted.append(int(element))
                    else:
                        element = round(element, 4)
                        converted.append(element)
                subview._data[
                    slice(
                        subview._offset,
                        subview._offset + calc_size(subview._shape) * step_size,
                        step_size,
                    )
                ] = converted
                idx += calc_size(subview._shape)
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])
        assert idx == len(values)

    def _flat(self):
        values = []
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            step_size = get_step_size(subview)
            if step_size:
                values += self._data[
                    slice(
                        subview._offset,
                        subview._offset + calc_size(subview._shape) * step_size,
                        step_size,
                    )
                ]
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])
        return values

    @property
    def flat(self):
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            step_size = get_step_size(subview)
            if step_size:
                for dim in self._data[
                    slice(
                        subview._offset,
                        subview._offset + calc_size(subview._shape) * step_size,
                        step_size,
                    )
                ]:
                    yield dim
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])
