from matplotlib import cm as plt_cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class __cm:
    def __init__(self):
        self.default = self.tab10

    def __getattr__(self, attr):
        return self.add_transparent_background(plt_cm.get_cmap(attr))

    def add_transparent_background(self, cmap, N=256):
        cmap = cmap(range(N))
        cmap = ListedColormap([(0, 0, 0, 0), *cmap])
        return cmap


cm = __cm()

