import numpy as np


def rotation2d_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    M = np.array([[c, -s],
                  [s, c]])
    return M