
    def format_repr(self, s, axis, offset, pad=0, whitespace=0, only=False):
        if only:
            return str(self._data[0])
        indent = min(2, max(0, (len(self._shape) - axis - 1)))
        if axis < len(self._shape):
            s += "["
            for idx in range(self._shape[axis]):
                if idx > 0:
                    s += ("\n " + " " * pad + " " * axis) * indent
                _oset = offset + idx * self._strides[axis] // self._itemsize
                s = self.format_repr(s, axis + 1, _oset, pad, whitespace)
                if idx < self._shape[axis] - 1:
                    s += ", "
            s += "]"
        else:
            r = repr(self._data[offset])
            if "." in r and r.endswith(".0"):
                r = f"{r[:-1]:<{whitespace}}"
            else:
                r = f"{r:>{whitespace}}"
            s += r
        return s

    def __repr__(self) -> str:
        size = calc_size(self._shape)
        whitespace = max(len(str(self._data[idx])) for idx in range(size))
        only = len(self._data) == 1
        formatted = self.format_repr("", 0, self._offset, 6, whitespace, only)
        if (
            self._dtype != float32
            and self._dtype != int32
            and self._dtype != bool
        ):
            return f"array({formatted}, dtype={self._dtype.numpy})"
        else:
            return f"array({formatted})"
