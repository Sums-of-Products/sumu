import numpy as np


class Data:
    """Class for holding data.

    The data can be input as either a path to a space delimited csv file, a
    numpy array or an object of type :py:class:`.Data` (in which case a new
    object is created pointing to the same underlying data).

    Assumes the input data is either discrete or continuous. The type
    is either read directly from the input (numpy array or Data), or
    it is inferred from the file the input path points to: "." is considered
    a decimal separator, i.e., it indicates continuous data.
    """

    def __init__(self, data_or_path):

        # Copying existing Data object
        if type(data_or_path) is Data:
            self.data = data_or_path.data
            self.discrete = data_or_path.discrete
            self.data_path = data_or_path.data_path
            return

        # Initializing from np.array
        if type(data_or_path) is np.ndarray:
            # TODO: Should cast all int types to np.int32 as that is what bdeu
            # scorer expects. Also make sure float is np.float64, not
            # np.float32 or so?
            self.data_path = None
            self.data = data_or_path
            self.discrete = self.data.dtype != np.float64
            return

        # Initializing from path
        if type(data_or_path) is str:
            self.data_path = data_or_path
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
            raise TypeError(f"Unknown type for Data: {type(data_or_path)}.")

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
    def shape(self):
        return self.data.shape

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
        info = dict()
        if self.data_path:
            info["data file"] = self.data_path
        info.update(
            {
                "no. variables": self.n,
                "no. data points": self.N,
                "type of data": ["continuous", "discrete"][1 * self.discrete],
            }
        )
        if self.discrete:
            info[
                "arities [min, max]"
            ] = f"[{min(self.arities)}, {max(self.arities)}]"
        return info

    def __repr__(self):
        return self.data.__repr__()

    def __getitem__(self, index):
        return self.data.__getitem__(index)
