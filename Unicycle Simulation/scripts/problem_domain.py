import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
import scipy.signal as signal
from geometry import rotation2d_matrix
import sys
sys.path.insert(0, '../utility')
from utility.matrixmath import mdot, sympart


# Time-correlated, time-varying, state-dependent disturbance
# This is a more direct LTI state-space formulation of filtered noise
# TODO Unused for now
def update_disturbance(t, Ts, x, w, Aw=None, Bw=None, z_mean=None, z_covr=None, z_dist='gaussian'):
    # w: n-dimensional random vector with time-correlation i.e. "colored noise"
    # z: d-dimensional random vector without time-correlation i.e. "white noise"
    n = w.size
    d = np.copy(n)
    angle = x[2]
    V = rotation2d_matrix(angle)
    # Linear filter data
    if Aw is None:
        # Apply greater disturbance momentum in the rolling direction than the transverse direction
        # Choose between [-1, 0], more negative means more decay/less momentum
        rolling_decay = -0.2/Ts
        transverse_decay = -0.8/Ts
        p_decay = mdot(V, np.diag([rolling_decay, transverse_decay]), V.T)
        steering_decay = -0.5/Ts
        Awc = sla.block_diag(p_decay, steering_decay)
        Aw = np.eye(n) + Awc*Ts
    if Bw is None:
        Bwc = np.eye(n)
        Bw = Bwc*Ts
    if z_mean is None:
        z_mean = np.zeros(d)
    if z_covr is None:
        # Apply greater noise in the rolling direction than the transverse direction
        rolling_var = 1.0
        transverse_var = 0.1
        p_covr = mdot(V, np.diag([rolling_var, transverse_var]), V.T)
        var_steering = 0.1
        z_covr = 0.01*sla.block_diag(p_covr, var_steering)
    # Generate the white noise
    if z_dist == 'gaussian':
        z = npr.multivariate_normal(z_mean, z_covr)
    elif z_dist == 'rademacher':
        z = z_mean + np.dot(sla.sqrtm(z_covr), 2*npr.binomial(1, 0.5, 3)-1)
    # Return the colored noise
    w_new = np.dot(Aw, w) + np.dot(Bw, z)
    return w_new


# This and the following functions are part of a less direct frequency-domain filtered noise scheme
def generate_filtered_noise(filter_order, filter_freq, fs, T, filter_type='butter', distribution='gaussian'):
    if filter_type == 'butter':
        filter_function = signal.butter
    elif filter_type == 'bessel':
        filter_function = signal.bessel
    else:
        raise Exception('Invalid filter type selected, ' +
                        'please choose a valid Matlab-style IIR filter design function from the SciPy package.')
    b, a = filter_function(filter_order, filter_freq, fs=fs)
    if distribution == 'gaussian':
        x = npr.randn(T)
    y = signal.filtfilt(b, a, x, method='gust')  # Note that this doubles the filter order
    return y


def generate_disturbance_hist(T, Ts, scale=1.0):
    fs = 1/Ts
    w_hist_rolling = scale*0.5*generate_filtered_noise(filter_order=1, filter_freq=0.01, fs=fs, T=T)
    w_hist_transverse = scale*0.01*generate_filtered_noise(filter_order=2, filter_freq=0.1, fs=fs, T=T)
    w_hist_steering = scale*0.05*generate_filtered_noise(filter_order=2, filter_freq=0.05, fs=fs, T=T)
    w_base_hist = np.vstack([w_hist_rolling, w_hist_transverse, w_hist_steering]).T
    return w_base_hist


# State-dependent disturbance
def transform_disturbance(w_base, x):
    angle = x[2]
    V = rotation2d_matrix(angle)
    w01 = np.copy(np.dot(V, w_base[0:2]))
    w2 = np.copy(w_base[2])
    return np.hstack([w01, w2])