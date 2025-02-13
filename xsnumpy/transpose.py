    def _view(self, shape, strides):
        out = self.__class__.__new__(self.__class__)
        out._shape = shape
        out._strides = strides
        out._data = self._data
        out._dtype = self._dtype
        out._offset = self._offset
        out._itemsize = self._itemsize
        return out

    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(reversed(range(len(self._shape))))
        elif sorted(axes) != list(range(len(self._shape))):
            raise ValueError("Invalid axes permutation")
        shape = tuple(self._shape[axis] for axis in axes)
        strides = tuple(self._strides[axis] for axis in axes)
        return self._view(shape=shape, strides=strides)
