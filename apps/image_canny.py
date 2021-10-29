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
from skimage import feature
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
sigma = st.sidebar.slider("sigma", min_value=0.0, max_value=10.0, value=1.0)
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
        edges = feature.canny(img, sigma=sigma)
        if show_segm:
            segm = find_1d_peaks(img)
        # img = skimage.transform.rotate(img, -angle, resize=True)
        # mask = skimage.transform.rotate(mask, -angle, resize=True)
        # mask = np.fft.fftshift(np.fft.fft2(img))
        fig = plt.figure()
        plt.imshow(img, cmap="gray")
        plt.imshow(edges, alpha=0.5, cmap=cm.vessel)
        col1.pyplot(fig)
        fig = plt.figure()
        # plt.title("porcine aorta")
        # plt.imshow(img, cmap="gray")
        plt.imshow(img, cmap="gray")
        plt.imshow(edges, alpha=0.5, cmap=cm.vessel)
        if show_segm:
            plt.imshow(segm, cmap=cm.rb, alpha=0.5)
        col2.pyplot(fig)
        plt.close("all")
        break
