    @property
    def strides(self):
        return self._strides

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return calc_size(self._shape)

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    @property
    def base(self):
        return self._base

    @property
    def dtype(self):
        return self._dtype
