#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script to run and test functions in `lqr.py`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Ben Gravell
Email:
benjamin.gravell@utdallas.edu
Github:
@BenGravell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script does the following:
- Provides sample trajectories used for debugging
- Provide functions to generate filtered noise
- Runs lqr and lqrm functions from `lqr.py` using a sample trajectory as the high-level trajectory
(edit `use_robust_lqr` to switch between lqr and lqrm)


Tested platform:
- Python 3.6.9 on Ubuntu 18.04 LTS (64 bit)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
import scipy.signal as signal
import matplotlib.pyplot as plt

from problem_domain import generate_disturbance_hist, transform_disturbance, update_disturbance
from plotting import plot_hist, plot_gain_hist, animate, plot_paths
from lqr import lqr, lqrm
from utility.matrixmath import mdot, sympart

from copy import copy
import time
import pickle
from opt_path import load_pickle_file
import os
import sys
sys.path.insert(0, '../utility')

import config
STEER_TIME = config.STEER_TIME  # Maximum Steering Time Horizon
DT = config.DT  # timestep between controls
SAVEPATH = config.SAVEPATH
GOALAREA = config.GOALAREA  #[xmin,xmax,ymin,ymax]
VELMIN, VELMAX = config.VELMIN, config.VELMAX
ANGVELMIN, ANGVELMAX = config.ANGVELMIN, config.ANGVELMAX
OBSTACLELIST = config.OBSTACLELIST
ROBRAD = config.ROBRAD
QLL = config.QLL
RLL = config.RLL
QTLL = config.QTLL


# State
# x[0] = horizontal position
# x[1] = vertical position
# x[2] = angular position
#
# Input
# u[0] = linear speed
# u[1] = angular speed

class OpenLoopController:
    def __init__(self, u_ref_hist):
        self.u_ref_hist = u_ref_hist

    def compute_input(self, x, t):
        u_ref = self.u_ref_hist[t]
        return u_ref


class LQRController:
    def __init__(self, K_hist, L_hist, e_hist, z_hist, x_ref_hist, u_ref_hist):
        self.K_hist = K_hist
        self.L_hist = L_hist
        self.e_hist = e_hist
        self.z_hist = z_hist
        self.x_ref_hist = x_ref_hist
        self.u_ref_hist = u_ref_hist

    def compute_input(self, x, t):
        K = self.K_hist[t]
        L = self.L_hist[t]
        e = self.e_hist[t]
        z = self.z_hist[t]
        x_ref = self.x_ref_hist[t]
        u_ref = self.u_ref_hist[t]
        dx = x - x_ref
        u = np.dot(K, dx) + np.dot(L, z) + e + u_ref
        return u


def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = load_pickle_file(input_file)
    ref_states = load_pickle_file(states_file)
    return ref_states, ref_inputs


def rollout(n, m, T, DT, x0=None, w_base_hist=None, controller=None, saturate_inputs=True, disturb=True, transform_disturbance_flag=True):
    # Initialize
    x_hist = np.zeros([T, n])
    if x0 is None:
        x0 = np.zeros(n)
    x_hist[0] = x0
    x = np.copy(x0)
    u_hist = np.zeros([T, m])
    w_hist = np.zeros([T, n])
    # Simulate
    for t in range(T-1):
        # Compute desired control inputs
        u = controller.compute_input(x, t)
        # Saturate inputs at actuator limits
        if saturate_inputs:
            u[0] = np.clip(u[0], VELMIN, VELMAX)
            u[1] = np.clip(u[1], ANGVELMIN, ANGVELMAX)
        # Generate state-dependent additive disturbance
        if disturb:
            if transform_disturbance_flag:
                w_base = w_base_hist[t]
                w = transform_disturbance(w_base, x)
            else:
                w = w_base_hist[t]
        else:
            w = np.zeros(n)
        # Transition the state
        x = dtime_dynamics(x, u, DT) + w
        # Record quantities
        x_hist[t+1] = x
        u_hist[t] = u
        w_hist[t+1] = w
    return x_hist, u_hist, w_hist


# Continuous-time nonlinear dynamics
def ctime_dynamics(x, u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])


# Discrete-time nonlinear dynamics
def dtime_dynamics(x, u, DT):
    # Euler method
    return x + ctime_dynamics(x, u)*DT


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
def dtime_jacobian(n, x, u, DT, method='zoh'):
    A, B = ctime_jacobian(x, u)
    Ad = np.eye(n) + A*DT
    Bd = B*DT
    # C, D = np.eye(n), np.zeros([n, m])
    # sysd = signal.cont2discrete((A, B, C, D), DT, method)
    # return sysd[0], sysd[1]
    return Ad, Bd


# generate some example reference inputs
# this is replaced by RRT* which actually does something useful
def generate_reference_inputs(pattern='rounded_arrow'):
    u_hist = np.zeros([T, m])
    for i in range(T):
        t = t_hist[i]
        if pattern == 'rounded_arrow':
            u_hist[i] = np.array([0.1*(2+0.5*np.cos(0.2*t)), 0.08*np.sin(0.05*t)])
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


def create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True,
                           delta_theta_max=1*(2*np.pi/360)):
    # delta_theta_max is the maximum assumed angle deviation in radians for robust control design

    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape
    # Compute linearized dynamics matrices along the reference trajectory
    A_hist = np.zeros([T, n, n])
    B_hist = np.zeros([T, n, m])
    for t in range(T):
        A_hist[t], B_hist[t] = dtime_jacobian(n, x_ref_hist[t], u_ref_hist[t], DT)

    E_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful
    W_hist = np.zeros([T, n, n])  # TODO - make this something more meaningful

    # Construct multiplicative noises and additive adversary

    # TODO - move this somewhere else
    # Old robustness settings
    # c = 3
    # C_hist = np.zeros([T, n, c])
    # for t in range(T):
    #     # Adversary can push robot around isotropically in xy plane position and twist the robot angle a little
    #     C_hist[t] = np.array([[0.4, 0.0, 0.0],
    #                           [0.0, 0.4, 0.0],
    #                           [0.0, 0.0, 0.1]])
    #
    # num_alphas = 3
    # num_betas = 2
    # num_gammas = 2
    # alpha_var = robust_scale*0.1*np.array([1.0, 1.0, 0.5])
    # beta_var = robust_scale*0.5*np.array([1.0, 0.5])
    # gamma_var = np.array([0, 0])
    # alpha_var_hist = np.tile(alpha_var, (T, 1))
    # beta_var_hist = np.tile(beta_var, (T, 1))
    # gamma_var_hist = np.tile(gamma_var, (T, 1))
    #
    # Ai_hist = np.zeros([T, num_alphas, n, n])
    # Bi_hist = np.zeros([T, num_betas, n, m])
    # Ci_hist = np.zeros([T, num_gammas, n, c])
    # for t in range(T):
    #     cos_theta = np.cos(x_ref_hist[t, 2])
    #     sin_theta = np.sin(x_ref_hist[t, 2])
    #     Ai_hist[t, 0] = np.array([[cos_theta, 0, 0],
    #                               [sin_theta, 0, 0],
    #                               [0, 0, 0]])
    #     Ai_hist[t, 1] = np.array([[0, cos_theta, 0],
    #                               [0, sin_theta, 0],
    #                               [0, 0, 0]])
    #     Ai_hist[t, 2] = np.array([[0, 0, 0],
    #                               [0, 0, 0],
    #                               [0, 0, 1]])
    #     Bi_hist[t, 0] = np.array([[cos_theta,  0],
    #                               [sin_theta,  0],
    #                               [0,  0]])
    #     Bi_hist[t, 1] = np.array([[0, 0],
    #                               [0, 0],
    #                               [0, 1]])
    #

    # New robustness settings
    c = 3
    C_hist = np.zeros([T, n, c])
    for t in range(T):
        # No adversary
        C_hist[t] = np.zeros([n, c])

    num_alphas = 2
    num_betas = 2
    num_gammas = 2
    alpha_var_hist = np.zeros([T, num_alphas])
    beta_var_hist = np.zeros([T, num_betas])
    gamma_var_hist = np.zeros([T, num_gammas])

    Ai_hist = np.zeros([T, num_alphas, n, n])
    Bi_hist = np.zeros([T, num_betas, n, m])
    Ci_hist = np.zeros([T, num_gammas, n, c])
    for t in range(T):
        v = u_ref_hist[t, 0]
        sin_theta = np.sin(x_ref_hist[t, 2])
        cos_theta = np.cos(x_ref_hist[t, 2])

        sin_delta_theta_max = np.sin(delta_theta_max)
        cos_delta_theta_max = np.cos(delta_theta_max)

        alpha_var_hist[t, 0] = v*DT*sin_delta_theta_max
        alpha_var_hist[t, 1] = v*DT*(1 - cos_delta_theta_max)
        beta_var_hist[t, 0] = sin_delta_theta_max
        beta_var_hist[t, 1] = 1 - cos_delta_theta_max

        Ai_hist[t, 0] = np.array([[0, 0, -sin_theta],
                                  [0, 0, cos_theta],
                                  [0, 0, 0]])
        Ai_hist[t, 1] = np.array([[0, 0, -cos_theta],
                                  [0, 0, -sin_theta],
                                  [0, 0, 0]])
        Bi_hist[t, 0] = np.array([[-sin_theta,  0],
                                  [cos_theta,  0],
                                  [0,  0]])
        Bi_hist[t, 1] = np.array([[cos_theta,  0],
                                  [sin_theta,  0],
                                  [0,  0]])

    # Construct cost matrices
    # We use the same cost matrices for all time steps, including the final time
    Qorg = np.diag([0, 0, 0])    # Penalty on state being far from origin
    # Qref = np.diag([100, 100, 10])  # Penalty on state deviating from reference
    # QTref = np.diag([100, 100, 10])
    Qref = QLL
    QTref = QTLL
    # Rorg = np.diag([1, 10])    # Penalty on control being far from origin (control effort)
    Rorg = RLL
    Rref = np.diag([0, 0])       # Penalty on input deviating from reference (deviation control effort)
    # Vorg = (1/robust_scale)*600*np.diag([2, 2, 1])       # Penalty on additive adversary
    Vorg = 1000 * np.diag([1, 1, 1])  # Penalty on additive adversary


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

    # Terminal penalty
    G_hist[-1] = np.block([[QTref+Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rref+Rorg, Zmc, Zmn, Rorg],
                              [Zcn, Zcm, -Vorg, Zcn, Zcm],
                              [Qorg, Znm, Znc, Qorg, Znm],
                              [Zmn, Rorg, Zmc, Zmn, Rorg]])

    # Construct the exogenous signal
    z_hist = np.hstack([x_ref_hist, u_ref_hist])

    # Compute optimal control policies, backwards in time
    if not use_robust_lqr:
        # Truncate the G_hist[k] to match the expected format of lqr() i.e. with no adversary blocks
        G_hist_for_lqr = np.zeros([T, n+m+n+m, n+m+n+m])
        for t in range(T):
            Znm, Zmn = np.zeros([n, m]), np.zeros([m, n])
            G_hist_for_lqr[t] = np.block([[Qref+Qorg, Znm, Qorg, Znm],
                                  [Zmn, Rref+Rorg, Zmn, Rorg],
                                  [Qorg, Znm, Qorg, Znm],
                                  [Zmn, Rorg, Zmn, Rorg]])
        K_hist, L_hist, e_hist, P_hist, q_hist, r_hist = lqr(z_hist, A_hist, B_hist, G_hist_for_lqr)
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

    return LQRController(K_hist, L_hist, e_hist, z_hist, x_ref_hist, u_ref_hist)


if __name__ == "__main__":
    # # Open-loop control sequence
    # T = 500
    # t_hist = np.arange(T)*DT
    # n, m = 3, 2
    # u_ref_hist = generate_reference_inputs(pattern='rounded_arrow')
    # # Create open-loop controller object
    # ol_controller = OpenLoopController(u_ref_hist)
    # # Get reference trajectory by simulating open-loop control using nonlinear dynamics, forwards in time
    # x_ref_hist, u_ref_hist, w_ref_hist = rollout(n, m, T, DT, x0=np.array([-3, -4, 0]), w_base_hist=None, controller=ol_controller,
    #                                           saturate_inputs=True, disturb=False)

    # input_file = "OptTraj_short_v1_0_1607441105_inputs"
    # input_file = "OptTraj_v1_0_1607307033_inputs"
    input_file = 'OptTraj_short_v1_0_1614486007_inputs'

    x_ref_hist, u_ref_hist = load_ref_traj(input_file)

    # Start in the reference initial state
    x0 = x_ref_hist[0]
    # Start in a different initial state to stress-test controllers
    # If only using linearization about reference trajectory, this may lead to catastrophic failure
    # since the actual trajectory will be different and thus the dynamics different and instability may result
    # x0 = np.array([-1, -1, 0.5])

    # Number of states, inputs
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape

    t_hist = np.arange(T) * DT

    # Create open-loop controller object
    ol_controller = OpenLoopController(u_ref_hist)

    # Create LQR controller object
    lqr_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=False)
    lqrm_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True)


    # Monte Carlo
    num_montecarlo_trials = 20
    x_cl_hist_all = np.zeros([num_montecarlo_trials, T, n])
    x_ol_hist_all = np.zeros([num_montecarlo_trials, T, n])
    for i in range(num_montecarlo_trials):

        w_base_hist = generate_disturbance_hist(T, DT, scale=0.2)

        # Simulate trajectory with noise and control, forwards in time
        x_cl_hist, u_cl_hist, w_cl_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=lqrm_controller,
                                                  saturate_inputs=True, disturb=True)

        x_ol_hist, u_ol_hist, w_ol_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=ol_controller,
                                                  saturate_inputs=True, disturb=True)

        x_cl_hist_all[i] = x_cl_hist
        x_ol_hist_all[i] = x_ol_hist

        # # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
        # Qeval = np.diag([10, 10, 1])
        # Reval = np.diag([10, 100])
        # Seval = np.diag([10, 10, 1])
        # print('Evaluation under closed-loop control')
        # evaluate_trajectory(T, x_cl_hist, u_cl_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        # print('')
        # print('Evaluation under open-loop control')
        # evaluate_trajectory(T, x_ol_hist, u_ol_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        # print('')


    # Plotting
    plt.close('all')

    # Plot Monte Carlo paths
    ax_lim = [-5.2, 5.2, -5.2, 5.2]
    x_hist_all_dict = {'Open-loop': x_ol_hist_all, 'Closed-loop': x_cl_hist_all}
    plot_paths(t_hist, x_hist_all_dict, {}, x_ref_hist, title=None, fig_offset=None,  axis_limits=ax_lim)





    # plot_hist(t_hist, [x_ref_hist, x_ol_hist, x_cl_hist], quantity='state')
    # plot_hist(t_hist, [u_ref_hist, u_ol_hist, u_cl_hist], quantity='input')
    # # plot_hist(t_hist, [w_ref_hist, w_ol_hist, w_cl_hist], quantity='disturbance')
    # plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')
    # plot_gain_hist(lqrm_controller.K_hist)

    # ax_lim = [-6, 6, -6, 6]
    # robot_w = 0.2/2
    # robot_h = 0.5/2
    # wheel_w = 0.5/2
    # wheel_h = 0.005/2
    #
    # animate(t_hist, x_ref_hist, u_ref_hist, x_ref_hist, u_ref_hist, title='Open-loop, reference', fig_offset=(400, 400),
    #         axis_limits = ax_lim, robot_w = robot_w, robot_h = robot_h, wheel_w = wheel_w, wheel_h = wheel_h)
    # animate(t_hist, x_cl_hist, u_cl_hist, x_ref_hist, u_ref_hist, title='Closed-loop, disturbed', fig_offset=(1000, 400),
    #         axis_limits = ax_lim, robot_w = robot_w, robot_h = robot_h, wheel_w = wheel_w, wheel_h = wheel_h)
    # animate(t_hist, x_ol_hist, u_ol_hist, x_ref_hist, u_ref_hist, title='Open-loop, disturbed', fig_offset=(1600, 400),
    #         axis_limits = ax_lim, robot_w = robot_w, robot_h = robot_h, wheel_w = wheel_w, wheel_h = wheel_h)
