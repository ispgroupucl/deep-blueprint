import numpy as np
from scipy.interpolate import interpn
from numba import njit
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@njit(nogil=True)
def get_rot_grid(volume, new_basis):
    s = volume.shape
    xx, yy, zz = np.mgrid[
        -s[0] // 2 : s[0] // 2, -s[1] // 2 : s[1] // 2, -s[2] // 2 : s[2] // 2
    ]
    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()
    grid = np.stack([xx, yy, zz])
    rot_grid = np.linalg.solve(new_basis, grid)
    return rot_grid


def rotate_basis(volume, new_basis):
    s = volume.shape
    rot_grid = get_rot_grid(volume, new_basis)
    rot_vol = interpn(
        (
            np.arange(-s[0] // 2, s[0] // 2),
            np.arange(-s[1] // 2, s[1] // 2),
            np.arange(-s[2] // 2, s[2] // 2),
        ),
        volume,
        rot_grid.T,
        bounds_error=False,
    ).reshape(s)

    return rot_vol


@njit(cache=True, nogil=True)
def compute_angle(vector):
    """Returns the angle in radians for a given vector."""
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


def orientation_3d(crop, num=100):
    ft = np.fft.fftn(crop, axes=(-3, -2, -1))
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
    confidence = pca.explained_variance_ratio_[0]  # TODO: compute
    return pca.components_, confidence


def rotate_crop3d(crop_shape, loc, volume, basis):
    """ Get rotated crop from complete volume.
        
        Inputs:
            crop_shape: 3-tuple for crop shape
            loc: location of center inside volume
            basis: matrix with values of the new x,y,z coordinate system
        Returns:
            rot_crop: rotated crop taken from volume
    """
    # step 1 : create crop grid
    sc = crop_shape
    xxc, yyc, zzc = np.mgrid[
        -sc[0] // 2 : sc[0] // 2, -sc[1] // 2 : sc[1] // 2, -sc[2] // 2 : sc[2] // 2
    ]
    xxc, yyc, zzc = xxc.flatten(), yyc.flatten(), zzc.flatten()
    crop_grid = np.stack([xxc, yyc, zzc])
    # step 2 : rotate cropped grid to new coordinate system
    crop_grid_rot = np.linalg.solve(basis, crop_grid)
    # step 3 : move grid to correct location
    crop_grid_rot = (crop_grid_rot.T + loc).T  # check if this is really correct
    # step 5 : interpolate
    s = volume.shape
    rot_crop = interpn(
        (np.arange(0, s[0]), np.arange(0, s[1]), np.arange(0, s[2])),
        volume,
        crop_grid_rot.T,
        bounds_error=True,
    ).reshape(sc)

    return rot_crop
