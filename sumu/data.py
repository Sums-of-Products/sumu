import numpy as np


class Data:
    """Class for holding data.

    The data can be input as either a path to a space delimited csv
    file, a numpy array or an object of type :py:class:`.Data` (in which case a new
    object is created pointing to the same underlying data).

    Assumes the input data is either discrete or continuous. The type
    is either read directly from the input (numpy array or Data), or
    it is inferred from the file the input path points to: "." is considered
    a decimal separator, i.e., it indicates continuous data.
    """

    def __init__(self, data_or_path):

        # Copying existing Data object
        if type(data_or_path) == Data:
            self.data = data_or_path.data
            self.discrete = data_or_path.discrete
            return

        # Initializing from np.array
        if type(data_or_path) == np.ndarray:
            self.data = data_or_path
            self.discrete = self.data.dtype != np.float64
            return

        # Initializing from path
        if type(data_or_path) == str:
            with open(data_or_path) as f:
                # . is assumed to be a decimal separator
                if "." in f.read():
                    self.discrete = False
                else:
                    self.discrete = True
            if self.discrete:
                self.data = np.loadtxt(
                    data_or_path, dtype=np.int32, delimiter=" "
                )
            else:  # continuous
                self.data = np.loadtxt(
                    data_or_path, dtype=np.float64, delimiter=" "
                )
            return

        else:
            raise TypeError(
                "Unknown type for Data: {}.".format(type(data_or_path))
            )

    def write_file(self, filepath):
        fmt = "%i" if self.discrete else "%f"
        np.savetxt(filepath, self.data, delimiter=" ", fmt=fmt)

    @property
    def n(self):
        return self.data.shape[1]

    @property
    def N(self):
        return self.data.shape[0]

    @property
    def arities(self):
        return np.count_nonzero(np.diff(np.sort(self.data.T)), axis=1) + 1

    def all(self):
        # TODO: NEED TO GET RID OF THIS?
        # This is to simplify passing data to R
        data = self.data
        if self.arities is not False:
            arities = np.reshape(self.arities, (-1, len(self.n)))
            data = np.append(arities, data, axis=0)
        return data

    @property
    def info(self):
        info = {
            "no. variables": self.n,
            "no. data points": self.N,
            "type of data": ["continuous", "discrete"][1 * self.discrete],
        }
        if self.discrete:
            info["arities [min, max]"] = "[{}, {}]".format(
                min(self.arities), max(self.arities)
            )
        return info

    def __repr__(self):
        return self.data.__repr__()

    def __getitem__(self, index):
        return self.data.__getitem__(index)
