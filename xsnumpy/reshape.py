    def reshape(self, shape):
        out = self.view()
        try:
            out.shape = shape
        except AttributeError:
            out = self.copy()
            out.shape = shape
        return out

    def view(self, dtype=None):
        if dtype is None:
            dtype = self._dtype
        if dtype == self._dtype:
            return ndarray(
                self._shape,
                dtype,
                buffer=self,
                offset=self._offset,
                strides=self._strides,
            )
        elif self.ndim == 1:
            itemsize = int(dtype.short[-1])
            size = calc_size(self._shape)
            offset = (self._offset * self._itemsize) // itemsize
            return ndarray(size, dtype, buffer=self, offset=offset)
        else:
            raise ValueError("Arrays can only be viewed with the same dtype")

    def astype(self, dtype):
        out = ndarray(self._shape, dtype)
        out[:] = self
        return out

    def copy(self):
        return self.astype(self._dtype)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value == self._shape:
            return
        if calc_size(self._shape) != calc_size(value):
            raise ValueError("New shape is incompatible with the current size")
        if get_step_size(self) == 1:
            self._shape = tuple(value)
            self._strides = calc_strides(self._shape, self._itemsize)
            return
        shape = [dim for dim in self._shape if dim > 1]
        strides = [
            stride for dim, stride in zip(self._shape, self._strides) if dim > 1
        ]
        new_shape = [dim for dim in value if dim > 1]
        if new_shape != shape:
            raise AttributeError(
                "New shape is incompatible with the current memory layout"
            )
        shape.append(1)
        strides.append(strides[-1])
        new_strides = []
        idx = len(shape) - 1
        for dim in reversed(value):
            if dim == 1:
                new_strides.append(strides[idx] * shape[idx])
            else:
                idx -= 1
                new_strides.append(strides[idx])
        if idx != -1:
            raise AttributeError(
                "New shape is incompatible with the current memory layout"
            )
        self._shape = tuple(value)
        self._strides = tuple(reversed(new_strides))
