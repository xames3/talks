class ndarray:

    def __init__(self, shape, dtype, buffer=None, offset=0, strides=None):
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = dtype
        self._itemsize = int(dtype.short[-1])
        self._offset = offset
        if buffer is None:
            self._base = None
            self._strides = calc_strides(self._shape, self._itemsize)
        else:
            if isinstance(buffer, ndarray) and buffer._base is not None:
                buffer = buffer._base
            self._base = buffer
            if isinstance(buffer, ndarray):
                buffer = buffer._data
            self._strides = calc_strides(self._shape, self._itemsize)
        buffersize = self._strides[0] * self._shape[0] // self._itemsize
        buffersize += self._offset
        Buffer = dtype[-1] * buffersize
        if buffer is None:
            if not isinstance(Buffer, str):
                self._data = Buffer()
        elif isinstance(buffer, ctypes.Array):
            self._data = Buffer.from_address(ctypes.addressof(buffer))
        else:
            self._data = Buffer.from_buffer(buffer)

