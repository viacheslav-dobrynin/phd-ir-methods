class LazyMap:
    def __init__(self, builder) -> None:
        self._builder = builder
        self._result = None

    def _get(self):
        if self._result is None:
            self._result = self._builder()
        return self._result

    def __getitem__(self, k):
        return self._get()[k]

    def __iter__(self):
        return iter(self._get())

    def __len__(self):
        return len(self._get())

    def __getattr__(self, name):
        return getattr(self._get(), name)
