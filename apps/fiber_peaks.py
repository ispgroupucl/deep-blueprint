from bioblue.fibers.fibers import find_1d_peaks
from pathlib import Path

import skimage.transform
import streamlit as st
import numpy as np
from bioblue import fibers
from bioblue.plot import cm
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.pipeline import make_pipeline
from numba import njit
import os

try:
    if __file__ is not None:
        st.set_page_config(layout="wide")
except NameError:
    pass


@njit(cache=True, nogil=True)
def compute_angle(vector):
    """ Returns the angle in radians between given vectors"""
    v1_u = vector / np.linalg.norm(vector)
    v2_u = np.array([1.0, 0.0])
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        sign = 1
    else:
        sign = np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return np.degrees(sign * np.arccos(dot_p))


# def compute_angle(component):
#     return np.degrees(np.arccos(np.clip(np.dot([1, 0], component), -1.0, 1.0)))


def chosen(ip):
    chosen_numbers = [
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 1),
        (3, 3),
        (3, 4),
        (3, 6),
        (4, 5),
    ]
    chosen_strings = [f"_{c[0]}_{c[1]}" for c in chosen_numbers]
    for c in chosen_strings:
        if c in str(ip):
            return True
    return False


ds_path = Path("../data/PA_fibers_cropped256/train")
img_paths = sorted((ds_path / "image").iterdir())
do_filter = st.sidebar.checkbox("Filter images?", value=True)
if do_filter:
    img_paths = [ip for ip in img_paths if chosen(ip)]
do_all = st.sidebar.checkbox("Run on all images ?", value=False)
if do_all:
    img_names = [ip.name for ip in img_paths]
else:
    img_paths = [ip.name for ip in img_paths]
    img_names = st.sidebar.multiselect("choose sample", img_paths)
find_1d_peaks = st.cache(fibers.find_1d_peaks)
angle = st.sidebar.slider("rotation", min_value=0, max_value=90, step=15)
minimum = st.sidebar.slider(
    "minimum", min_value=8.0, max_value=12.0, step=0.125, value=9.0
)
show_segm = st.sidebar.checkbox("Show fiber segmentation")
for img_name in img_names:
    img_path = ds_path / "image" / img_name
    img_npz = np.load(img_path)
    col1, col2 = st.beta_columns(2,)
    residues = st.empty()
    for i, name in enumerate(img_npz):
        img = img_npz[name]
        # img = img > 130
        img = skimage.transform.rotate(img, angle, resize=True) * 255
        if show_segm:
            segm = find_1d_peaks(img)
        # img = skimage.transform.rotate(img, -angle, resize=True)
        # mask = skimage.transform.rotate(mask, -angle, resize=True)
        # mask = np.fft.fftshift(np.fft.fft2(img))
        mask = np.fft.fft2(img)
        mask[0, :] = 1
        mask[:, 0] = 1
        mask = np.fft.fftshift(mask)
        fig = plt.figure()
        # plt.imshow(img, cmap="gray")
        mask_norm = np.log(np.abs(mask))
        mask_filtered = mask_norm.copy()
        mask_filtered[mask_filtered <= minimum] = 0
        data = list(np.nonzero(mask_norm > minimum))
        data[0] = data[0] - 128
        data[1] = data[1] - 128
        line = np.linspace(10, img.shape[1] - 10, num=255) - 128
        # data = scale(data, with_std=True, axis=1)
        if False:  # angle == 90:
            reg = LinearRegression().fit(data[1].reshape(-1, 1), data[0])
        else:
            reg = LinearRegression().fit(
                data[1].reshape(-1, 1), data[0]
            )  # , sample_weight=mask_norm[data])
        pca = make_pipeline(
            StandardScaler(with_mean=True, with_std=False), PCA(n_components=2)
        )
        pca = pca.fit(np.vstack(list(reversed(data))).T)
        pca: PCA = pca.named_steps["pca"]
        result = reg.predict(line.reshape(-1, 1))
        residues.text(reg._residues / len(data))
        pca_angle = compute_angle(pca.components_[0])
        pca_2_angle = compute_angle(pca.components_[1])
        plt.title(f"FFT norm {pca_angle:.2f} {pca_2_angle:.2f}")
        # plt.scatter(data[0, :], data[1, :])
        plt.imshow(mask_filtered, cmap="gray")
        if False:  # angle == 90:
            plt.plot(line, result)
        else:
            pass
            # plt.plot(result + 128, line + 128)  # line, result)
        a = np.degrees(np.arccos(np.dot(pca.components_[0], pca.components_[1])))
        for comp, var in zip(pca.components_, pca.explained_variance_):
            plt.plot(
                np.array([0, var * comp[0] / 5]) + 128,
                np.array([0, var * comp[1] / 5]) + 128,
            )
        col1.pyplot(fig)
        # plt.colorbar()
        # segm = segm_npz[name]
        # plt.imshow(segm == 1, cmap=cm.vessel, alpha=0.4)
        # col1.pyplot(fig)

        fig = plt.figure()
        plt.title("porcine aorta")
        # plt.imshow(img, cmap="gray")
        plt.imshow(img, cmap="gray")
        if show_segm:
            plt.imshow(segm, cmap=cm.vessel, alpha=0.5)
        for comp, var in zip(pca.components_, pca.explained_variance_):
            plt.plot(
                np.array([0, var * comp[0] / 5]) + 128,
                np.array([0, var * comp[1] / 5]) + 128,
            )
        # segm = segm_npz[name]
        # plt.imshow(segm == 1, cmap=cm.vessel, alpha=0.4)
        col2.pyplot(fig)
        plt.close("all")
        break
