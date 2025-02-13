    def broadcast_to(self, size):
        if self._shape == size:
            return self
        if len(size) < len(self._shape):
            raise ValueError(f"Cannot broadcast {self._shape} to {size}")
        data = self._data[:]
        for idx in range(len(size)):
            if idx >= len(self._shape) or self._shape[idx] == 1:
                data = data * size[idx]
        out = ndarray(size, self._dtype)
        out._data = data
        return out

    def __add__(self, other):
        out: ndarray
        if isinstance(other, (int, float)):
            dtype = (
                float32
                if isinstance(other, float)
                or self._dtype.numpy.startswith("float")
                else int32
            )
            out = ndarray(self._shape, dtype)
            out[:] = [data + other for data in self._data]
        elif isinstance(other, ndarray):
            dtype = (
                float32
                if self._dtype.numpy.startswith("float")
                or other._dtype.numpy.startswith("float")
                else int32
            )
            shape = broadcast_shape(self._shape, other._shape)
            self = self.broadcast_to(shape)
            other = other.broadcast_to(shape)
            out = ndarray(shape, dtype)
            out[:] = [x + y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out
