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
from scipy.ndimage import sobel
from scipy.ndimage.filters import convolve, maximum_filter
from skimage.filters import scharr_h, scharr_v, sobel_h, sobel_v, scharr

# from shapely.geometry import Polygon
from numba import njit

st.set_page_config(layout="wide")

ds_path = Path("../data/PA_fibers/train/image")
for volume_path in ds_path.iterdir():
    volume = np.load(volume_path)
    imagename = st.sidebar.select_slider("slice", volume.files)
    # grid_size = st.sidebar.slider(
    #     "grid size", min_value=20, max_value=400, value=200, step=20
    # )
    grid_size = 20
    conv_size = st.sidebar.slider("kernel size", min_value=3, max_value=100)
    image = volume[imagename]
    orig_fig = plt.figure()
    gX = convolve(
        scharr(image, axis=0), np.ones((conv_size, conv_size)) / (conv_size ** 2)
    )  # sobel(image, axis=0)
    gY = convolve(
        scharr(image, axis=1), np.ones((conv_size, conv_size)) / (conv_size ** 2)
    )
    orientation = np.arctan2(gY, gX, dtype=float) * (180 / np.pi) % 180
    orientation = convolve(orientation, np.ones((10, 10)) / (10 * 10))
    magnitude = np.sqrt((gX ** 2) + (gY ** 2), dtype=float)
    mag_filtered = maximum_filter(magnitude, size=20)
    # orientation[magnitude != mag_filtered] = -1
    st.write(magnitude.mean())
    orientation.shape
    orientation.dtype
    xx, yy = np.mgrid[
        0 : image.shape[1] : grid_size, 0 : image.shape[0] : grid_size,
    ]
    # xx = xx.flatten()
    # yy = yy.flatten()
    # xx.shape
    # yy.shape
    # image_h[::100, ::100].shape
    # plt.imshow(image, cmap="gray")
    # plt.quiver(
    #     xx,
    #     yy,
    #     image_v[::grid_size, ::grid_size],
    #     -image_h[::grid_size, ::grid_size],
    #     color="red",
    # )
    plt.imshow(orientation)  # , alpha=m)
    plt.colorbar()
    plt.show()
    st.pyplot(orig_fig)
    break
