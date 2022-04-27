#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
import scipy.signal as signal
import matplotlib.pyplot as plt
from plotting import plot_hist, plot_gain_hist, animate
from lqr import lqr, lqrm
from utility.matrixmath import mdot, sympart
from copy import copy

# State
# x[0] = horizontal position
# x[1] = vertical position
# x[2] = angular position
#
# Input
# u[0] = linear speed
# u[1] = angular speed


# Continuous-time nonlinear dynamics
def ctime_dynamics(x, u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])


# Discrete-time nonlinear dynamics
def dtime_dynamics(x, u, Ts):
    # Euler method
    return x + ctime_dynamics(x, u)*Ts


# Linearized continuous-time dynamics
def ctime_jacobian(x, u):
    A = np.array([[0, 0, -u[0]*np.sin(x[2])],
                  [0, 0, u[0]*np.cos(x[2])],
                  [0, 0, 0]])
    B = np.array([[np.cos(x[2]), 0],
                  [np.sin(x[2]), 0],
                  [0, 1]])
    return A, B


# Linearized discrete-time dynamics
def dtime_jacobian(n, x, u, Ts, method='zoh'):
    A, B = ctime_jacobian(x, u)
    Ad = np.eye(n) + A*Ts
    Bd = B*Ts
    # C, D = np.eye(n), np.zeros([n, m])
    # sysd = signal.cont2discrete((A, B, C, D), Ts, method)
    # return sysd[0], sysd[1]
    return Ad, Bd


def rotation2d_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    M = np.array([[c, -s],
                  [s, c]])
    return M


# Time-correlated, time-varying, state-dependent disturbance
# This is a more direct LTI state-space formulation of filtered noise
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


def rollout(n, m, T, Ts, x0=None, w0=None, w_base_hist=None, u_hist=None,
            K_hist=None, L_hist=None, e_hist=None, x_ref_hist=None, u_ref_hist=None, z_hist=None,
            closed_loop=True, disturb=True):
    # Initialize
    x_hist = np.zeros([T, n])
    if x0 is None:
        x0 = np.zeros(n)
    x_hist[0] = x0
    if u_hist is None:
        u_hist = np.zeros([T, m])
    w_hist = np.zeros([T, n])
    if w0 is not None:
        w_hist[0] = w0
    # Simulate
    for t in range(T-1):
        if K_hist is not None:
            K = K_hist[t]
        if L_hist is not None:
            L = L_hist[t]
        if e_hist is not None:
            d = e_hist[t]
        x = x_hist[t]
        if z_hist is not None:
            z = z_hist[t]
        if x_ref_hist is not None:
            x_ref = np.copy(x_ref_hist[t])
        if u_ref_hist is not None:
            u_ref = np.copy(u_ref_hist[t])
        if closed_loop:
            dx = x - x_ref
            u = np.dot(K, dx) + np.dot(L, z) + d + u_ref
        else:
            u = u_hist[t]
        # Saturate inputs at actuator limits
        saturate_inputs = False
        if saturate_inputs:
            u0_min, u0_max = -0.5, 0.5
            u1_min, u1_max = -0.1, 0.1
            u[0] = np.clip(u[0], u0_min, u0_max)
            u[1] = np.clip(u[1], u1_min, u1_max)
        if disturb:
            if w_base_hist is not None:
                w_base = w_base_hist[t]
                w = transform_disturbance(w_base, x)
            else:
                w = w_hist[t]
                w = update_disturbance(t, Ts, x, w)
        else:
            w = np.zeros(n)
        x_hist[t+1] = dtime_dynamics(x, u, Ts) + w
        u_hist[t] = u
        w_hist[t+1] = w
    return x_hist, u_hist, w_hist


def generate_reference_inputs(pattern='rounded_arrow'):
    u_hist = np.zeros([T, m])
    for i in range(T):
        t = t_hist[i]
        if pattern == 'rounded_arrow':
            u_hist[i] = np.array([0.1, 0.019*np.tanh(4*np.sin(0.01*t)**10)])
        elif pattern == 'clover':
            u_hist[i] = np.array([0.01*(np.sin(0.2*t)+1)+0.05*(np.sin(0.05*t)+1)+0.05,
                                  0.03*np.sin(0.05*t) + 0.02*np.tanh(4*np.sin(0.01*t)+1) - 0.005])
    return u_hist


def evaluate_trajectory(T, x_hist, u_hist, w_hist, x_ref_hist, Q, R, S):
    # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
    dxtot = 0
    utot = 0
    wtot = 0
    for t in range(T):
        dx = x_hist[t] - x_ref_hist[t]
        u = u_hist[t]
        w = w_hist[t]
        dxtot += mdot(dx.T, Q, dx)
        utot += mdot(u.T, R, u)
        wtot += mdot(w.T, S, w)
    print('Total     tracking error: %.3f' % dxtot)
    print('Total     control effort: %.3f' % utot)
    print('Total disturbance energy: %.3f' % wtot)
    return dxtot, utot, wtot


if __name__ == "__main__":
    npr.seed(3)
    # Number of states, inputs
    n, m = 3, 2
    # Time start, end
    t0, tf = 0, 1000
    # Sampling period
    Ts = 0.5
    # Time history
    t_hist = np.arange(t0, tf, Ts)
    # Number of time steps
    T = t_hist.size
    # Initial state and disturbance
    x0 = np.array([0, 0, 0])
    # w0 = np.array([0, 0, 0])
    # Generate base disturbance sequence
    w_base_hist = generate_disturbance_hist(T, Ts, scale=0.5)
    E_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful
    W_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful

    # Open-loop control sequence
    u_ref_hist = generate_reference_inputs(pattern='rounded_arrow')

    # Get reference trajectory by simulating open-loop control using nonlinear dynamics, forwards in time
    x_ref_hist, u_ref_hist, w_ref_hist = rollout(n, m, T, Ts, x0=x0, w_base_hist=w_base_hist, u_hist=u_ref_hist,
                                                 closed_loop=False, disturb=False)

    # Compute linearized dynamics matrices along the reference trajectory
    A_hist = np.zeros([T, n, n])
    B_hist = np.zeros([T, n, m])
    for t in range(T):
        A_hist[t], B_hist[t] = dtime_jacobian(n, x_ref_hist[t], u_ref_hist[t], Ts)

    # Construct multiplicative noises and additive adversary
    c = 3
    C_hist = np.zeros([T, n, c])
    for t in range(T):
        # Adversary can push robot around isotropically in xy plane position and twist the robot angle a little
        C_hist[t] = np.array([[0.4, 0.0, 0.0],
                              [0.0, 0.4, 0.0],
                              [0.0, 0.0, 0.1]])

    num_alphas = 3
    num_betas = 2
    num_gammas = 2
    alpha_var = 0.1*np.array([1.0, 1.0, 0.5])
    beta_var = 0.5*np.array([1.0, 0.5])
    gamma_var = np.array([0, 0])
    alpha_var_hist = np.tile(alpha_var, (T, 1))
    beta_var_hist = np.tile(beta_var, (T, 1))
    gamma_var_hist = np.tile(gamma_var, (T, 1))

    Ai_hist = np.zeros([T, num_alphas, n, n])
    Bi_hist = np.zeros([T, num_betas, n, m])
    Ci_hist = np.zeros([T, num_gammas, n, c])
    for t in range(T):
        cos_theta = np.cos(x_ref_hist[t, 2])
        sin_theta = np.sin(x_ref_hist[t, 2])
        Ai_hist[t, 0] = np.array([[cos_theta, 0, 0],
                                  [sin_theta, 0, 0],
                                  [0, 0, 0]])
        Ai_hist[t, 1] = np.array([[0, cos_theta, 0],
                                  [0, sin_theta, 0],
                                  [0, 0, 0]])
        Ai_hist[t, 2] = np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 1]])
        Bi_hist[t, 0] = np.array([[cos_theta,  0],
                                  [sin_theta,  0],
                                  [0,  0]])
        Bi_hist[t, 1] = np.array([[0, 0],
                                  [0, 0],
                                  [0, 1]])

    # Construct cost matrices
    # We use the same cost matrices for all time steps, including the final time
    Qorg = np.diag([0, 0, 0])    # Penalty on state being far from origin
    Qref = np.diag([10, 10, 1])  # Penalty on state deviating from reference
    Rorg = np.diag([10, 100])    # Penalty on control being far from origin (control effort)
    Rref = np.diag([0, 0])       # Penalty on input deviating from reference (deviation control effort)
    Vorg = 600*np.diag([2, 2, 1])       # Penalty on additive adversary

    G_hist = np.zeros([T, n+m+c+n+m, n+m+c+n+m])
    for t in range(T):
        Znm, Zmn = np.zeros([n, m]), np.zeros([m, n])
        Znc, Zmc = np.zeros([n, c]), np.zeros([m, c])
        Zcn, Zcm = np.zeros([c, n]), np.zeros([c, m])

        G_hist[t] = np.block([[Qref+Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rref+Rorg, Zmc, Zmn, Rorg],
                              [Zcn, Zcm, -Vorg, Zcn, Zcm],
                              [Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rorg, Zmc, Zmn, Rorg]])

    # Construct the exogenous signal
    z_hist = np.hstack([x_ref_hist, u_ref_hist])

    # Compute optimal control policies, backwards in time
    use_robust_lqr = True
    if not use_robust_lqr:
        K_hist, L_hist, e_hist, P_hist, q_hist, r_hist = lqr(z_hist, A_hist, B_hist, G_hist)
    else:
        lqrm_args = {'z_hist': z_hist,
                     'A_hist': A_hist,
                     'B_hist': B_hist,
                     'C_hist': C_hist,
                     'Ai_hist': Ai_hist,
                     'Bi_hist': Bi_hist,
                     'Ci_hist': Ci_hist,
                     'alpha_var_hist': alpha_var_hist,
                     'beta_var_hist': beta_var_hist,
                     'gamma_var_hist': gamma_var_hist,
                     'G_hist': G_hist,
                     'E_hist': E_hist,
                     'W_hist': W_hist}
        lqrm_outs = lqrm(**lqrm_args)
        K_hist, L_hist, e_hist, Kv_hist, Lv_hist, ev_hist, P_hist, q_hist, r_hist = lqrm_outs

    # Start in a different initial state to stress-test controllers
    # If only using linearization about reference trajectory, this may lead to catastrophic failure
    # since the actual trajectory will be different and thus the dynamics different and instability may result
    # x0 = np.array([-1, -1, 0.5])

    # Simulate trajectory with noise and closed-loop control, forwards in time
    x_cl_hist, u_cl_hist, w_cl_hist = rollout(n, m, T, Ts, x0=x0, w_base_hist=w_base_hist,
                                              K_hist=K_hist, L_hist=L_hist, e_hist=e_hist,
                                              x_ref_hist=x_ref_hist, u_ref_hist=u_ref_hist, z_hist=z_hist,
                                              closed_loop=True, disturb=True)

    # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
    Qeval = np.diag([10, 10, 1])
    Reval = np.diag([10, 100])
    Seval = np.diag([10, 10, 1])
    evaluate_trajectory(T, x_cl_hist, u_cl_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)

    # Simulate trajectory with noise and open-loop control, forwards in time
    x_ol_hist, u_ol_hist, w_ol_hist = rollout(n, m, T, Ts, x0=x0, w_base_hist=w_base_hist, u_hist=u_ref_hist,
                                              closed_loop=False, disturb=True)

    plt.close('all')
    plot_hist(t_hist, [x_ref_hist, x_ol_hist, x_cl_hist], quantity='state')
    plot_hist(t_hist, [u_ref_hist, u_ol_hist, u_cl_hist], quantity='input')
    plot_hist(t_hist, [w_ref_hist, w_ol_hist, w_cl_hist], quantity='disturbance')
    plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')
    plot_gain_hist(K_hist)

    animate(t_hist, x_ref_hist, u_ref_hist, x_ref_hist, u_ref_hist, title='Open-loop, reference', fig_offset=(400, 400))
    animate(t_hist, x_cl_hist, u_cl_hist, x_ref_hist, u_ref_hist, title='Closed-loop, disturbed', fig_offset=(1000, 400))
    animate(t_hist, x_ol_hist, u_ol_hist, x_ref_hist, u_ref_hist, title='Open-loop, disturbed', fig_offset=(1600, 400))
