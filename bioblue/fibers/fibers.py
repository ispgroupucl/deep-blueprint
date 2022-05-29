from typing_extensions import final
import igraph as ig
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numba import jit, njit, prange
from itertools import repeat
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree, kdtree
import scipy.ndimage as ndi
import skimage.measure.profile as profile
from bioblue.plot import cm
from enum import Enum


def find_direction(img, percentile=75):
    plt.imshow(img > np.percentile(img, percentile))
    plt.show()
    x, y = np.nonzero(img > np.percentile(img, percentile))
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    return evecs[:, sort_indices[0]], evecs[:, sort_indices[1]]


def find_1d_peaks(img, rel_height=0.5, height=110, flat=False, segment_valleys=False):
    if flat:
        fiber_points = []
    else:
        fiber_points = np.zeros_like(img)
    for i in range(img.shape[1]):
        x = img[:, i]
        peaks, _ = signal.find_peaks(x, height=height)
        if flat:
            fiber_points += zip(peaks, repeat(i))  # , x[peaks])
        else:
            fiber_points[peaks, i] = 1
            if segment_valleys:
                valleys, _ = signal.find_peaks(255 - x)
                fiber_points[valleys, i] = 2

    return fiber_points


def find_peaks_in_volume(
    vol, rel_height=0.5, height=110, flat=False, segment_valleys=False
):
    fiber_points = np.zeros_like(vol)
    xx, yy = np.mgrid[0 : vol.shape[1], 0 : vol.shape[2]]
    xx, yy = xx.flatten(), yy.flatten()
    for i, j in zip(xx, yy):
        x = vol[:, i, j]
        peaks, _ = signal.find_peaks(x, height=height)
        fiber_points[peaks, i, j] = 1
        if segment_valleys:
            valleys, _ = signal.find_peaks(255 - x)
            fiber_points[valleys, i, j] = 2
    return fiber_points


def find_peaks_along_lines(img, height=110, fix_x=True, fix_y=False):
    assert not (fix_x == True and fix_y == True)
    if fix_x:
        x_list = repeat(0), repeat(img.shape[0])
    else:
        x_list = range(img.shape[0]), range(img.shape[0])
    if fix_y:
        y_list = repeat(0), repeat(img.shape[1])
    else:
        y_list = range(img.shape[1]), range(img.shape[1])
    mask = np.zeros_like(img)
    srcs = list(zip(x_list[0], y_list[0]))
    dsts = list(zip(x_list[1], y_list[1]))
    for src, dst in zip(srcs, dsts):
        print(src, dst)
        x = profile.profile_line(img, src=src, dst=dst)
        peaks, _ = signal.find_peaks(x, height=height)
        line = profile._line_profile_coordinates(src, dst)
        if len(peaks) > 0:
            peaks_on_line = line[:, peaks, 0].astype(np.int)
            mask[peaks_on_line[0], peaks_on_line[1]] = 1
    return mask


def find_1d_valleys(img):
    valley_points = np.zeros_like(img)
    for i in range(img.shape[1]):
        x = img[:, i]
        valleys, _ = signal.find_peaks(255 - x)
        valley_points[valleys, i] = 1


def compute_distances(peaks):
    distances = squareform(pdist(peaks))
    return distances


@njit(parallel=True)
def compute_sparse_pi_distances(peaks, img, simple_pdist, max_radius=10):
    data = []
    rows = []
    cols = []
    for i in prange(peaks.shape[0]):
        for j in prange(peaks.shape[0]):
            if simple_pdist[i, j] > max_radius:
                continue
            start_peak = peaks[i]
            end_peak = peaks[j]
            xx, yy = line_nd(start_peak, end_peak)
            intensities = np.zeros(len(xx))
            for idx, (x, y) in enumerate(zip(xx, yy)):
                intensities[idx] = img[x, y]
            data.append(np.sum(np.abs(np.diff(intensities))))
            rows.append(i)
            cols.append(j)

    return data, rows, cols


@njit(parallel=True)
def compute_pi_distances(peaks, img, simple_pdist, max_radius=10):
    pdists = np.ones((peaks.shape[0], peaks.shape[0])) * np.inf
    for i in prange(peaks.shape[0]):
        for j in prange(peaks.shape[0]):
            if simple_pdist[i, j] > max_radius:
                continue
            start_peak = peaks[i]
            end_peak = peaks[j]
            xx, yy = line_nd(start_peak, end_peak)
            intensities = np.zeros(len(xx))
            for idx, (x, y) in enumerate(zip(xx, yy)):
                intensities[idx] = img[x, y]
            pdists[i, j] = (
                np.mean(np.abs(np.diff(intensities))) + 2 * simple_pdist[i, j]
            )

    return pdists


def plot_lines(peaks, img, simple_pdist, max_radius=10):
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[0]):
            if simple_pdist[i, j] > max_radius:
                continue
            start_peak = peaks[i]
            end_peak = peaks[j]
            xx, yy = line_nd(start_peak, end_peak)
            line_img = np.zeros_like(img)
            line_img[xx, yy] = 1
            intensities = np.zeros(len(xx))
            for idx, (x, y) in enumerate(zip(xx, yy)):
                intensities[idx] = img[x, y]

            plt.title(f"sum-diff-intensities {np.sum(np.abs(np.diff(intensities)))}")
            plt.imshow(img, cmap="gray")
            plt.scatter([start_peak[1], end_peak[1]], [start_peak[0], end_peak[0]])
            plt.imshow(line_img, cmap=cm.vessel, alpha=0.7)
            plt.show()


@njit()
def _round_safe(coords):
    """Round coords while ensuring succesive values are less than 1 apart.
    
    See : https://github.com/scikit-image/scikit-image/blob/2bbfce4786102a3a9e85839a52303e858346a08e/skimage/draw/draw_nd.py#L4
    """
    if len(coords) > 1 and coords[0] % 1 == 0.5 and coords[1] - coords[0] == 1:
        return np.floor(coords).astype(np.int32)
    else:
        out = np.empty_like(coords)
        np.round(coords, 0, out)
        return out.astype(np.int32)


@njit()
def line_nd(start, stop):
    """Draw a single-pixel thick line in n dimensions.
    
    See : https://github.com/scikit-image/scikit-image/blob/2bbfce4786102a3a9e85839a52303e858346a08e/skimage/draw/draw_nd.py#L54
    """
    start = np.asarray(start)
    stop = np.asarray(stop)
    npoints = int(np.ceil(np.max(np.abs(stop - start))))
    npoints += 1
    coords = np.empty((len(start), npoints))
    for i, (s, e) in enumerate(zip(start, stop)):
        coords[i, :] = np.linspace(s, e, npoints).T
    # coords = np.stack(coords)
    # print(coords, coords.shape)

    for dim in range(len(start)):
        coords[dim, :] = _round_safe(coords[dim, :])

    coords = coords.astype(np.int32)
    return coords


class FIFOSearch:
    def __init__(self, peak_img, peaks, img) -> None:
        self.peak_img = peak_img
        self.peaks = peaks
        self.peak_indices = np.ones_like(img, dtype=np.int32) * -1
        self.peak_indices[peaks[:, 0], peaks[:, 1]] = range(self.peaks.shape[0])
        self.img = img
        self.pdists = compute_distances(self.peaks)
        # plot_lines(peaks, img, self.pdists, max_radius=5)
        self.pidists = compute_pi_distances(self.peaks, img, self.pdists, max_radius=10)
        self.sorted_pidists = np.argsort(self.pidists, axis=-1)
        self.indices_pidists = np.unravel_index(
            np.argsort(self.pidists, axis=None), self.pidists.shape
        )
        self.radius = 0
        self.print_statistics()

    def print_statistics(self):
        real_sorted_pidists = np.sort(self.pidists, axis=-1)
        print(
            real_sorted_pidists[:, 1:2].mean(),
            real_sorted_pidists[:, 1:2].max(),
            real_sorted_pidists[:, 1:2].min(),
        )

    def search(self, radius=10, min_length=10):
        fibers = []
        for index_i, index_j in zip(*self.indices_pidists):
            if index_i <= index_j:
                continue
            if self.pidists[index_i, index_j] > radius:
                break
            findex_i, findex_j = None, None
            for i, fiber in enumerate(fibers):
                if index_i in fiber[1:-1] or index_j in fiber[1:-1]:
                    break  # one of the fiber is not at an extremity
                if index_i in fiber and index_j in fiber:
                    break
                if index_i == fiber[0]:
                    findex_i = i, 0
                elif index_j == fiber[0]:
                    findex_j = i, 0
                elif index_i == fiber[-1]:
                    findex_i = i, len(fiber)
                elif index_j == fiber[-1]:
                    findex_j = i, len(fiber)
                elif index_i in fiber or index_j in fiber:
                    # assert False
                    break  # one of the fibers is not in an extremity
            else:  # if no break
                if findex_i is None and findex_j is None:
                    fibers.append([index_i, index_j])
                elif findex_i is None:
                    fibers[findex_j[0]].insert(findex_j[1], index_i)
                    assert fibers[findex_j[0]][findex_j[1]] == index_i
                elif findex_j is None:
                    fibers[findex_i[0]].insert(findex_i[1], index_j)
                    assert fibers[findex_i[0]][findex_i[1]] == index_j
                else:
                    if findex_i[0] > findex_j[0]:
                        fiber_i = fibers.pop(findex_i[0])
                        fiber_j = fibers.pop(findex_j[0])
                    else:
                        fiber_j = fibers.pop(findex_j[0])
                        fiber_i = fibers.pop(findex_i[0])
                    if findex_i[1] == 0:
                        fiber_i = list(reversed(fiber_i))
                    if findex_j[1] == -1:
                        fiber_j = list(reversed(fiber_j))
                    # print(fiber_i, fiber_j, findex_i, findex_j)
                    fibers.append(fiber_i + fiber_j)
                # print(
                #     self.peaks[index_i],
                #     self.peaks[index_j],
                #     self.pidists[index_i, index_j],
                #     fibers,
                # )
                # print()

        fiber_img = np.zeros_like(self.img, dtype=np.uint32)
        heads = []
        tails = []
        fiber_points = []
        i = 1
        for fiber in fibers:
            if len(fiber) < min_length:
                continue
            heads.append(self.peaks[fiber[0]])
            tails.append(self.peaks[fiber[-1]])
            assert list(self.peaks[fiber[0]]) != list(self.peaks[fiber[-1]]), f"{fiber}"
            fiber_points.append([])
            for point in fiber:
                x, y = self.peaks[point]
                fiber_points[-1].append((x, y))
                fiber_img[x, y] = i
            i += 1
        # print([len(fiber) for fiber in fibers])
        print(max([len(fiber) for fiber in fibers]))
        return (
            fiber_img,
            heads,
            tails,
            fiber_points,
        )


class GreedySearch:
    def __init__(self, peak_img, peaks, img) -> None:
        self.kdtree = KDTree(peaks)
        self.peak_img = peak_img
        self.peaks = peaks
        self.peak_indices = np.ones_like(img, dtype=np.int32) * -1
        self.peak_indices[peaks[:, 0], peaks[:, 1]] = range(self.peaks.shape[0])
        self.img = img
        self.pdists = compute_distances(self.peaks)
        self.pidists = compute_pi_distances(self.peaks, img, self.pdists, max_radius=10)
        self.sorted_pidists = np.argsort(self.pidists, axis=-1)
        self.real_sorted_pidists = np.sort(self.pidists, axis=-1)
        print(
            self.real_sorted_pidists[:, 1:2].mean(),
            self.real_sorted_pidists[:, 1:2].max(),
            self.real_sorted_pidists[:, 1:2].min(),
        )
        self.radius = 0
        # assert np.all(self.sorted_pdists[:,0] == np.array(range(self.peaks.shape[0])))

    def find_neighbour_intensity(self, row_idx, col_idx):
        peak_index = self.peak_indices[row_idx, col_idx]
        assert peak_index != -1
        column = self.sorted_pidists[peak_index]
        final_i = None
        for index in column[1:]:
            if self.pidists[peak_index, index] > self.radius:
                assert (
                    self.pidists[peak_index, index] != np.inf
                ), "You should augment max_radius"
                break
            if self.peaks[index, 1] >= col_idx:
                final_i = index
                break

        if final_i is not None:
            idx = self.peaks[final_i]
            return idx
        else:
            return None

    def find_neighbour(self, row_idx, col_idx):
        indices = self.kdtree.query_ball_point(
            (row_idx, col_idx), r=self.radius, return_sorted=True
        )
        final_i = None
        for i in indices:
            if self.peaks[i][1] > col_idx:
                final_i = i
                break
        if final_i is not None:
            idx = self.peaks[final_i]
            return idx
        else:
            return None

    def iterative_search(self, radius=2.1, step=10):
        columns = np.zeros_like(self.img)
        for start in range(0, self.peak_img.shape[1], step):
            partial_columns = self.search(start, radius)
            columns[partial_columns != 0] = partial_columns[partial_columns != 0]

        return columns

    def search(self, start=0, radius=2.1, number=1, direction=+1):
        self.radius = radius
        columns = np.copy(self.peak_img)
        columns[:, start + 1 :] = 0
        if start != 0:
            columns[:, 0:start] = 0

        # For loop :
        for col_idx in range(start, columns.shape[1]):
            # print(f"column {col_idx}")
            ii = 0
            for row_idx in range(columns.shape[0]):
                if columns[row_idx, col_idx] == 0:
                    continue
                ii += 1
                intensity = self.img[row_idx, col_idx]
                idx = self.find_neighbour_intensity(row_idx, col_idx)
                if idx is not None:
                    columns[idx[0], idx[1]] = 1

        return columns


def greedy_search(peak_img, peaks, img, start=0, radius=2.1, number=1, direction=+1):
    # Init :
    kdtree: KDTree = KDTree(peaks)
    columns = np.copy(peak_img)
    columns[:, start + 1 :] = 0
    if start != 0:
        columns[:, 0:start] = 0
    # For loop :
    for col_idx in range(columns.shape[1]):
        # print(f"column {col_idx}")
        ii = 0
        for row_idx in range(columns.shape[0]):
            if columns[row_idx, col_idx] == 0:
                continue
            ii += 1
            intensity = img[row_idx, col_idx]
            d, indices = kdtree.query(
                (row_idx, col_idx, intensity), k=5
            )  # , r=radius, return_sorted=True)
            final_i = None
            for i in indices:
                if peaks[i][1] > col_idx:
                    final_i = i
                    break
            if final_i is not None:
                idx = peaks[final_i]
                columns[idx[0], idx[1]] = 1

    return columns
