from pathlib import Path
import pandas as pd

# from shapely.geometry.point import Point
from skimage.filters.thresholding import threshold_multiotsu
from sklearn.pipeline import make_pipeline
import streamlit as st
import bioblue as bb
from bioblue import fibers
from bioblue.plot import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import colorcet as ct
from skimage.transform import rotate
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

# from shapely.geometry import Polygon
from numba import njit

st.set_page_config(layout="wide")


# @njit(cache=True, nogil=True)
def compute_angle(vector):
    """ Returns the angle in radians between given vectors"""
    v1_u = vector / np.linalg.norm(vector)
    v2_u = np.array([1.0, 0.0, 0.0])
    minor = np.linalg.det(np.stack((v1_u[-3:], v2_u[-3:])))
    if minor == 0:
        sign = 1
    else:
        sign = np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return np.degrees(sign * np.arccos(dot_p))


def orientation(crop, num=100):
    ft = np.fft.fftn(crop, axes=(-3, -2, -1))
    # Remove constant term
    ft[0, :, :] = 1
    ft[:, 0, :] = 1
    ft[:, :, 0] = 1
    ft = np.fft.fftshift(ft)
    ft_logabs = np.log(np.abs(ft))
    # data = np.nonzero(ft_logabs > minimum)
    data = np.unravel_index(np.argsort(-ft_logabs, axis=None)[:num], shape=crop.shape)
    ft_filtered = np.zeros_like(crop)
    ft_filtered[data[0], data[1], data[2]] = 1
    # ft_filtered = ft_logabs > minimum
    if len(data[0]) == 0:
        raise NotImplementedError()
    pca = make_pipeline(
        StandardScaler(with_mean=True, with_std=False), PCA(n_components=3)
    )
    crop_data = np.vstack(list(data)).T
    mean = np.mean(crop_data, axis=0)
    mean
    pca = pca.fit(crop_data)
    pca: PCA = pca.named_steps["pca"]
    # angle = compute_angle(pca.components_[0])
    return (
        None,
        pca.components_,
        (pca.explained_variance_, pca.explained_variance_ratio_),
        ft_filtered,
        crop_data,
    )


ds_path = Path("../data/PA_fibers/train/image")
for volume_path in ds_path.iterdir():
    volume = np.load(volume_path)

    grid_size = st.sidebar.slider(
        "grid size", min_value=20, max_value=400, value=200, step=20
    )
    crop_size = st.sidebar.slider(
        "crop size", min_value=20, max_value=400, value=100, step=10
    )
    zsize = len(volume.files)
    xsize, ysize = volume[volume.files[0]].shape
    xsize, ysize, zsize
    st.sidebar.write("### ROI")
    border = 400
    x0roi, x1roi = st.sidebar.slider(
        "x", 0, xsize, value=(border, xsize - border), step=10
    )
    y0roi, y1roi = st.sidebar.slider(
        "y", 0, ysize, value=(border, ysize - border), step=10
    )
    z0roi, z1roi = st.sidebar.slider(
        "z", 0, zsize, value=(border, zsize - border), step=10
    )
    xx, yy, zz = np.mgrid[
        x0roi + crop_size // 2 : x1roi - crop_size // 2 : grid_size,
        y0roi + crop_size // 2 : y1roi - crop_size // 2 : grid_size,
        z0roi + crop_size // 2 : z1roi - crop_size // 2 : grid_size,
    ]
    xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
    xx.shape, yy.shape, zz.shape
    st.sidebar.write("### Crop Position")
    x, y, z = st.sidebar.selectbox("position", zip(xx, yy, zz))

    crop_volume = np.zeros((crop_size, crop_size, crop_size))
    for i, file in enumerate(volume.files[z - crop_size // 2 : z + crop_size // 2]):
        image = volume[file]
        crop_image = image[
            x - crop_size // 2 : x + crop_size // 2,
            y - crop_size // 2 : y + crop_size // 2,
        ]
        crop_volume[:, :, i] = crop_image

    st.write(crop_volume.min(), crop_volume.max())
    fig = plt.figure()
    plt.imshow(crop_volume[:, :, 0])
    st.pyplot(fig, clear_figure=True)
    num = st.slider("Number of points for PCA", 10, 10000, value=100, step=10,)
    angle, components, variance, ft_filtered, crop_data = orientation(
        crop_volume, num=num
    )

    ft = np.fft.fftn(crop_volume, axes=(-3, -2, -1))
    # Remove constant frequencies
    ft[0, :, :] = 1
    ft[:, 0, :] = 1
    ft[:, :, 0] = 1
    ft = np.fft.fftshift(ft)
    ft = np.log(np.abs(ft))
    st.write(ft.min(), ft.max())

    xxcrop, yycrop, zzcrop = np.mgrid[0:crop_size, 0:crop_size, 0:crop_size]
    xxcrop, yycrop, zzcrop = xxcrop.flatten(), yycrop.flatten(), zzcrop.flatten()
    ft = ft.flatten()
    sample_size = len(ft)
    ft.shape
    perc = 100 - 100 * (500 / sample_size)
    plotly_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=crop_data[:, 0],
                y=crop_data[:, 1],
                z=crop_data[:, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.5, color="black"),
            ),
            go.Scatter3d(
                x=[50, 50 + 10 * components[0, 0]],
                y=[50, 50 + 10 * components[0, 1]],
                z=[50, 50 + 10 * components[0, 2]],
                line=dict(width=4, color="red"),
                mode="lines",
                marker=dict(size=0),
            ),
            go.Scatter3d(
                x=[50, 50 + 10 * components[1, 0]],
                y=[50, 50 + 10 * components[1, 1]],
                z=[50, 50 + 10 * components[1, 2]],
                line=dict(width=4, color="green"),
                mode="lines",
                marker=dict(size=0),
            ),
            go.Scatter3d(
                x=[50, 50 + 10 * components[2, 0]],
                y=[50, 50 + 10 * components[2, 1]],
                z=[50, 50 + 10 * components[2, 2]],
                line=dict(width=4, color="blue"),
                mode="lines",
                marker=dict(size=0),
            ),
        ]
    )
    st.plotly_chart(plotly_fig, use_container_width=True)
