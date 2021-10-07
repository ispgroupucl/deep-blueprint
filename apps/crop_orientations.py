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

# from shapely.geometry import Polygon
from numba import njit

st.set_page_config(layout="wide")


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


def rotate_and_crop(image, center, angle, size):
    image = rotate(image.copy(), angle, center=center) * 255
    return image[
        center[1] - size // 2 : center[1] + size // 2,
        center[0] - size // 2 : center[0] + size // 2,
    ]


def orientation(crop, num=100):
    ft = np.fft.fft2(crop)
    # Remove constant term
    ft[0, :] = 1
    ft[:, 0] = 1
    ft = np.fft.fftshift(ft)
    ft_logabs = np.log(np.abs(ft))
    # data = np.nonzero(ft_logabs > minimum)
    data = np.unravel_index(np.argsort(-ft_logabs, axis=None)[:num], shape=crop.shape)
    ft_filtered = np.zeros_like(crop)
    ft_filtered[data[0], data[1]] = 1
    # ft_filtered = ft_logabs > minimum
    if len(data[0]) == 0:
        raise NotImplementedError()
    pca = make_pipeline(
        StandardScaler(with_mean=True, with_std=False), PCA(n_components=2)
    )
    pca = pca.fit(np.vstack(list(reversed(data))).T)
    pca: PCA = pca.named_steps["pca"]
    angle = compute_angle(pca.components_[0])
    return (
        angle,
        pca.components_,
        (pca.explained_variance_, pca.explained_variance_ratio_),
        ft_filtered,
    )


ds_path = Path("../data/PA_fibers/train/image")
for volume_path in ds_path.iterdir():
    volume = np.load(volume_path)
    imagename = st.sidebar.select_slider("slice", volume.files)
    grid_size = st.sidebar.slider(
        "grid size", min_value=20, max_value=400, value=200, step=20
    )
    crop_size = st.sidebar.slider(
        "crop size", min_value=20, max_value=400, value=100, step=10
    )
    image = volume[imagename]
    orig_fig = plt.figure()
    plt.imshow(image, cmap="gray")
    st.pyplot(orig_fig)
    # with st.beta_expander("canvas"):
    #     canvas_result = st_canvas(
    #         fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    #         stroke_width=1,
    #         stroke_color="red",
    #         background_color="white",
    #         background_image=Image.fromarray(image),
    #         update_streamlit=True,
    #         height=image.shape[0] // 2,
    #         width=image.shape[1] // 2,
    #         drawing_mode="polygon",
    #         key="canvas",
    #     )

    xx, yy = np.mgrid[
        crop_size // 2 : image.shape[1] - crop_size // 2 : (grid_size // 2),
        crop_size // 2 : image.shape[0] - crop_size // 2 : (grid_size // 2),
    ]
    # if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
    #     points = []
    #     for point in canvas_result.json_data["objects"][0]["path"]:
    #         if len(point) > 1:
    #             points.append(np.array(point[1:]) * 2)
    #     points = Polygon(points)
    # else:
    points = None
    xx = xx.flatten()
    yy = yy.flatten()
    # col1, col2 = st.beta_columns(2)
    # st_crop = col1.empty()
    # idx = st.sidebar.slider("grid idx", min_value=0, max_value=len(xx) - 1)
    # x, y = xx[idx], yy[idx]
    # x, y = st.sidebar.select_slider("grid index", zip(xx, yy))
    fig_img, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    if points is not None:
        filtered_xx = []
        # ax.plot(*points.exterior.xy)
    num = st.sidebar.slider("num", min_value=10, max_value=200, step=10, value=50)
    orientations = np.zeros_like(image, dtype=np.float)
    residues = []
    show_var = st.sidebar.checkbox("Show PCA var")
    threshs = threshold_multiotsu(image, classes=4)
    thresh = threshs[0]
    st.text(thresh)
    fig_hist = plt.figure()
    plt.hist(image.flatten(), bins=255, range=(0, 255))
    plt.axvline(thresh, color="red")
    st.pyplot(fig_hist)
    for x, y in zip(xx, yy):
        # if points is not None and not points.contains(Point(x, y)):
        #     continue
        crop = image[
            y - crop_size // 2 : y + crop_size // 2,
            x - crop_size // 2 : x + crop_size // 2,
        ]
        if crop.min() < thresh:
            continue
        # fig = plt.figure()
        # plt.imshow(crop)
        # st_crop.pyplot(fig)

        angle, components, explained_var, ft_filtered = orientation(crop, num=num)
        orientations[
            y - crop_size // 2 : y + crop_size // 2,
            x - crop_size // 2 : x + crop_size // 2,
        ] = ft_filtered
        # if explained_var[0][0] < 50 or explained_var[1][0] < 0.9:
        #     continue
        # ax.text(x, y, f"{angle:0.0f}", color="red", size="x-small")
        if angle < 0:
            angle += 180
        rotated_crop = rotate_and_crop(image, (x, y), 90 - angle, crop_size)
        if rotated_crop.min() < thresh - 10:
            continue
        # st.text(angle)
        for comp, var, color, s in zip(
            components, explained_var, ["red", "blue"], [grid_size // 8, grid_size // 4]
        ):
            var = var if show_var else s
            if color == "red":
                continue
            c = plt_cm.hsv(angle / 180)
            ax.plot(
                np.array([0, var * comp[0]]) + x,
                np.array([0, var * comp[1]]) + y,
                linewidth=0.5,
                c=c,
            )
        residues.append(np.mean(explained_var))
        # fig_crop = plt.figure()
        # plt.imshow(rotated_crop)
        # plt.title(f"{explained_var}")
        # st.pyplot(fig_crop)
        # plt.close()
        # if l is not None:
        #     ax.plot(r + x, l + y)

    # ax.scatter(xx, yy, s=1, c="red")
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    st.pyplot(fig_img)
    # fig_orientations = plt.figure()
    # plt.imshow(orientations, cmap=cm.magma)
    # st.pyplot(fig_orientations)

    # fig_img = plt.figure()
    # plt.imshow(image, cmap="gray")
    # plt.imshow(orientations, cmap=cm.magma)
    # st.pyplot(fig_img)
    break
