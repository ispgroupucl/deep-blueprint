from bioblue.plot import cm
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import cv2
import json

st.set_page_config(layout="wide")

ds_path = Path("../data/Task202_KidneyVessels_RT/train/image")
threshold_path = ds_path.parent / "threshold.json"

if "thresholds" not in st.session_state:
    if threshold_path.exists():
        with threshold_path.open("r") as fp:
            st.session_state["thresholds"] = json.load(fp)
    else:
        st.session_state["thresholds"] = {"checked": []}

thresholds = st.session_state["thresholds"]
show_on_image = st.sidebar.checkbox("Show segmentation on image", value=True)
side_by_side = st.sidebar.checkbox("Side by side view", value=False)
sample_name = st.sidebar.selectbox("Sample", sorted(p.name for p in ds_path.iterdir()))
sample_path = ds_path / sample_name
if sample_name not in thresholds:
    thresholds[sample_name] = {}
volume = np.load(sample_path)
idx = st.sidebar.slider("Index", min_value=0, max_value=len(volume) - 1)

slice = volume[volume.files[idx]]
slice = cv2.resize(slice, dsize=tuple(np.array(slice.shape) // 2))
fig = plt.figure(figsize=(8, 1))
fig.tight_layout()
plt.hist(slice.flatten(), density=True, bins=255, range=(0, 255))
plt.axis("off")
fig.patch.set_alpha(0.0)
if str(idx) in thresholds[sample_name]:
    threshold_value = thresholds[sample_name][str(idx)]
else:
    threshold_value = 100
threshold = st.sidebar.slider(
    "Threshold", value=threshold_value, min_value=0, max_value=255
)
plt.axvline(threshold, color="red")
st.sidebar.pyplot(fig)
if st.sidebar.button("Set threshold"):
    thresholds[sample_name][str(idx)] = threshold
    # thresholds[sample_name].setdefault("checked", []).append(idx)
    thresholds[sample_name]["checked"] = sorted(
        list(set(thresholds[sample_name].get("checked", []) + [idx]))
    )
fig = plt.figure()
plt.tight_layout()
plt.imshow(slice, cmap="gray")
if show_on_image:
    plt.imshow(slice > threshold, cmap=cm.vessel, interpolation="none", alpha=0.5)
plt.axis("off")
if not side_by_side:
    layout = [3, 1]
else:
    layout = [1, 1]
col1, col2 = st.beta_columns(layout)
col1.pyplot(fig)
fig = plt.figure()
plt.imshow((slice > threshold) * 255, cmap=cm.vessel)
plt.axis("off")
col2.pyplot(fig)

if st.sidebar.button("Interpolate"):
    x = np.array(
        sorted(
            (int(k), int(v))
            for k, v in thresholds[sample_name].items()
            if k != "checked" and int(k) in thresholds[sample_name]["checked"]
        )
    )
    # x[:, 0]
    r = np.array(range(len(volume)))
    interp = np.interp(
        np.array(range(len(volume))), x[:, 0], x[:, 1], left=None, right=None
    ).astype(int)
    for k, v in enumerate(interp):
        if v is not None:
            thresholds[sample_name][str(k)] = int(v)
if st.sidebar.button("Remove interpolation"):
    for k in list(thresholds[sample_name].keys()):
        if k != "checked" and int(k) not in thresholds[sample_name]["checked"]:
            del thresholds[sample_name][k]

# thresholds
x = np.array([int(k) for k in thresholds[sample_name].keys() if k != "checked"])
order = np.argsort(x)
y = np.array([int(v) for k, v in thresholds[sample_name].items() if k != "checked"])
x = x[order]
y = y[order]
fig = plt.figure(figsize=(5, 1))
plt.plot(x, y)
plt.xlim([0, len(volume)])
plt.ylim([0, 255])
st.pyplot(fig)
if st.sidebar.button("Save"):
    with threshold_path.open("w") as fp:
        json.dump(thresholds, fp, indent=2)
