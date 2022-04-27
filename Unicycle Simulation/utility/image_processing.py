import numpy as np
from scipy.interpolate import RectBivariateSpline

# Low quality, fast processing downscaling
def downscale(x, scale):
    return x[::scale, ::scale]


# Low quality, fast processing upscaling
def upscale(x, scale):
    return np.repeat(np.repeat(x, scale, axis=0), scale, axis=1)


# Wrapper around RectBivariateSpline suitable for 2D images
def interpolate(z, scale, order=3):
    width, height = z.shape
    x, y = np.linspace(-1, 1, height), np.linspace(-1, 1, width)
    u, v = np.linspace(-1, 1, scale*height), np.linspace(-1, 1, scale*width)
    f = RectBivariateSpline(y, x, z, kx=order, ky=order)
    return f(v, u)
