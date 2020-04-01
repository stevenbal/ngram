from functools import reduce


class NestedDict(dict):
    """
    An extension of standard dictionaries that allows for automatic
    nested dictionaries
    """

    def __getitem__(self, key):
        """
        Description:    function that attempts to return the value at a
                        given key, if a KeyError occurs, the value at
                        that key is set to be an empty NestedDict

        Input:
        -key:           any hashable type, indicates the key for which
                        the value must be returned

        Output:
        -value:         any type, the value at the given key
        """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            value = self[key] = type(self)()
            return value

    def get_by_path(self, path):
        """
        Description:    function that returns the value given a list of
                        keys

        Input:
        -path:          list, the keys for which the value must be
                        retrieved from the NestedDict

        Output:
        -value:         any type, the value at the key path, if it does
                        not exist, 0 is returned instead
        """
        value = reduce(lambda x, y: x.get(y, {}), path, self)
        return value if value else 0

    def set_by_path(self, path, value):
        """
        Description:    function that sets a value at a given path of
                        keys

        Input:
        -path:          list, the keys at which the value must be set in
                        the NestedDict
        -value:         any type, the value that will be set at given
                        path
        """
        for key in path[:-1]:
            self = self.setdefault(key, {})
        self[path[-1]] = value

    def add_by_path(self, path, value):
        """
        Description:    function that adds a value at a given path of
                        keys

        Input:
        -path:          list, the keys at which the value must be added
                        in the NestedDict
        -value:         any type, the value that will added at the given
                        path
        """
        for key in path[:-1]:
            self = self.setdefault(key, {})
        if path[-1] in self:
            self[path[-1]] = self[path[-1]] + value
        else:
            self[path[-1]] = value
