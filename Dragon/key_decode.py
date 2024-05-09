import cloudpickle

class MyKey:
    import cloudpickle

    def __init__(self, f_name, hash_int):
        self._f_name = f_name
        self._hash_int = hash_int

    def __hash__(self):
        # look at self._f_name and decode on a integer to return that would always 
        # be the same for a given f_name.
        return int(self._hash_int)

    def __getstate__(self):
        return cloudpickle.dumps(self._f_name)