import numpy as np
from scipy import stats
from skimage import exposure


def kernel_3d_mv(mean, cov, size, num_planes):
    """ Create 3d gaussian kernel in (size, size, num_planes)
        Args:
            mean : 3-item array, containing x,y,z mean
            cov : 3x3 covariance matrix
            l : size of the 2d plane
            planes : number of planes
    """
    x, y, z = np.mgrid[0:size, 0:size, 0:num_planes]
    pos = np.empty(x.shape + (3,))
    pos[:, :, :, 0] = x
    pos[:, :, :, 1] = y
    pos[:, :, :, 2] = z
    result = stats.multivariate_normal.pdf(pos, mean, cov)
    return result


def kernel_3d(mean, sigma, size, num_planes):
    meanx, meany, meanz = mean
    sigx, sigy, sigz = sigma
    ax = np.linspace(0, size, size)
    axz = np.linspace(0, num_planes, num_planes)
    gaussx = np.exp(-0.5 * np.square(ax - meanx) / np.square(sigx))
    gaussy = np.exp(-0.5 * np.square(ax - meany) / np.square(sigy))
    gaussz = np.exp(-0.5 * np.square(axz - meanz) / np.square(sigz))
    kernel = np.multiply.outer(np.outer(gaussx, gaussy), gaussz)
    return kernel / np.sum(kernel)


def generate_planes(
    size: int,
    num_planes: int,
    num_distortions: int = 500,
    max_force=25,
    inter_plane_force=20,
):
    """ Create {num_planes} planes, with distortions.

        Args:
            size: size of the planes
            num_planes: number of planes
            num_distortions: number of gaussian kernels applied to volume 
        Returns:
            planes: a (size, size, num_planes) volume where intensity denotes
                    distortions.
    """
    rng = np.random.default_rng()
    planes_shape = (size, size, num_planes)
    planes = np.zeros(planes_shape)

    for _ in range(num_distortions):
        rmean = rng.integers(0, planes_shape, 3)
        force = 0.1  # rng.integers(1, max_force)
        plane_sigma = rng.random(2) * force * size
        inter_plane_sigma = 0.5  # rng.random() * inter_plane_force
        # sigma_start = rng.random() * force * np.array(planes_shape)
        # rsig = np.clip(sigma_start + rng.integers(-2, 2, 3), 0.01, None)
        ramp = rng.normal(0, 1) / force
        cov = np.diag(tuple(plane_sigma) + (inter_plane_sigma,))
        print(plane_sigma, inter_plane_sigma)
        kernel = kernel_3d(
            rmean, tuple(plane_sigma) + (inter_plane_sigma,), size, num_planes
        )
        planes += (ramp / np.max(kernel)) * kernel

    return planes


def generate_volume(planes, amplitude, radius):
    """ Generate (cube) volume from planes with distortions, 
    """
    rng = np.random.default_rng()
    size = planes.shape[0]
    num_planes = planes.shape[-1]
    freq = size // num_planes
    vol = rng.normal(0, 0.001, (size, size, size))
    starts = np.linspace(freq, size + 1, num_planes)

    for i, start in enumerate(starts):
        p = planes[:, :, i] * amplitude + start
        xx, yy = np.mgrid[0:size, 0:size]
        plane_r = rng.normal(0, 0.001)
        plane_d = rng.normal(0, 0.001)
        for x, y in zip(xx, yy):
            rand_radius = rng.normal(radius, 0.01) + plane_r
            zz = np.linspace(-rand_radius * 2, rand_radius * 2, 16)
            intensities = np.exp(-0.5 * np.square(zz) / np.square(rand_radius / 4))
            for z, intensity in zip(zz, intensities):
                pp = np.clip((p[x, y] + z + plane_d).astype(int), 0, size - 1)
                vol[pp, y, x] += intensity / np.sum(intensities) + rng.normal(0, 0.005)

    # Normalize & Equalize hist
    # vol = ((vol - np.min(vol)) / (np.max(vol) - np.min(vol))) * 255
    vol = exposure.equalize_hist(vol)
    return vol

