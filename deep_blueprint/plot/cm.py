from matplotlib import cm as plt_cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt


class __cm:
    vessel = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 1)])
    rgb = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    rb = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 1), (0, 0, 1, 1)])

    def __init__(self):
        self.default = self.tab10

    def __getattr__(self, attr):
        return self.add_transparent_background(plt_cm.get_cmap(attr))

    def add_transparent_background(self, cmap, N=None):
        if N is None:
            N = cmap.N
        if isinstance(cmap, ListedColormap):
            cmap = cmap(range(N))
            cmap = ListedColormap([(0, 0, 0, 0), *cmap])
        elif isinstance(cmap, LinearSegmentedColormap):
            cmap = cmap(range(N))
            cmap = LinearSegmentedColormap.from_list("cmap", [(0, 0, 0, 0), *cmap], N=N)
        else:
            raise TypeError("unknown type")
        return cmap


cm = __cm()

