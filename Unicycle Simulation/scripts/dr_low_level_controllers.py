#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script to run and test functions in `lqr.py`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

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
from scipy.stats import multivariate_normal
import scipy.linalg as sla
import scipy.signal as signal
import matplotlib.pyplot as plt

from problem_domain import generate_disturbance_hist, transform_disturbance, update_disturbance
from plotting import plot_hist, plot_gain_hist, animate # TODO: EDIT ANIMATE PLOTTING TOOL TO USE OBSTACLES FROM CONFIG FILE
from lqr import lqr, lqrm
from utility.matrixmath import mdot, sympart
from copy import copy
import time
import pickle
from opt_path import load_pickle_file
import os
import sys
import copy


sys.path.insert(0, '../utility')

from tracking_controller import dtime_dynamics, OpenLoopController, LQRController

from tracking_controller import rollout, ctime_dynamics, \
    ctime_jacobian, dtime_jacobian, generate_reference_inputs, evaluate_trajectory, create_lqrm_controller
from drrrts_nmpc import drrrtstar_with_nmpc

import config

STEER_TIME = config.STEER_TIME  # Maximum Steering Time Horizon
DT = config.DT  # timestep between controls
SAVEPATH = config.SAVEPATH
GOALAREA = config.GOALAREA  # [xmin,xmax,ymin,ymax]
VELMIN, VELMAX = config.VELMIN, config.VELMAX
ANGVELMIN, ANGVELMAX = config.ANGVELMIN, config.ANGVELMAX

OBSTACLELIST = copy.copy(config.OBSTACLELIST)
ROBRAD = config.ROBRAD

SIGMAW = config.SIGMAW  # Covariance of process noise

# State
# x[0] = horizontal position
# x[1] = vertical position
# x[2] = angular position
#
# Input
# u[0] = linear speed
# u[1] = angular speed


def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = load_pickle_file(input_file)
    ref_states = load_pickle_file(states_file)
    return ref_states, ref_inputs


def run_low_level_controller(run_lqr=False, run_robust_lqr=False, run_nmpc=False):
    # animation parameters
    ax_lim = [-6, 6, -6, 6]
    robot_w = 0.2 / 2
    robot_h = 0.5 / 2
    wheel_w = 0.5 / 2
    wheel_h = 0.005 / 2

    # steering params for NMPC
    v_max = VELMAX  # maximum linear velocity (m/s)
    omega_max = ANGVELMAX  # 0.125 * (2 * np.pi)  # maximum angular velocity (rad/s)
    x_max = 5  # maximum state in the horizontal direction
    y_max = 5  # maximum state in the vertical direction
    theta_max = np.inf  # maximum state in the theta direction
    nmpc_horizon = 10 # 60
    save_plot = False

    # load trajectory
    input_file = "OptTraj_short_v1_0_1614552961_inputs"
    x_ref_hist, u_ref_hist = load_ref_traj(input_file)
    inputs_string = "inputs"
    save_file_name = input_file.replace(inputs_string, "")

    # Start in the reference initial state
    x0 = x_ref_hist[0]

    # Number of states, inputs
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape

    t_hist = np.arange(T) * DT

    # create the disturbance
    SigmaW = SIGMAW
    sigma1 = SigmaW[0,0] # first entry in SigmaW
    dist = "nrm"  # "ben", "nrm", "lap", "gum"
    show_hist = False
    if dist == "ben":
        w_base_hist = generate_disturbance_hist(T, DT, scale=1)
        if show_hist:
            plt.hist(w_base_hist)
            plt.show()
    elif dist == "nrm":
        w_base_hist = npr.multivariate_normal(mean=[0, 0, 0], cov=SigmaW, size=T)
        if show_hist:
            plt.hist(w_base_hist)
            plt.show()
    elif dist == "lap":
        l = 0
        b = (sigma1 / 2) ** 0.5
        w_base_hist = npr.laplace(loc=l, scale=b, size=[T, 3]) # mean = loc, var = 2*scale^2
        if show_hist:
            plt.hist(w_base_hist)
            plt.show()
    elif dist == "gum":
        b = (6*sigma1)**0.5/np.pi
        l=-0.57721*b
        w_base_hist = npr.gumbel(loc=l, scale=b, size=[T,3]) # mean = loc+0.57721*scale, var = pi^2/6 scale^2
        if show_hist:
            plt.hist(w_base_hist)
            plt.show()

    # Create open-loop controller object
    ol_controller = OpenLoopController(u_ref_hist)

    ############# LQR #############
    if run_lqr:
        # Create LQR controller object
        lqrm_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=False)

        # Simulate trajectory with noise and control, forwards in time
        x_cl_hist, u_cl_hist, w_cl_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=lqrm_controller,
                                                  saturate_inputs=True, disturb=True)

        x_ol_hist, u_ol_hist, w_ol_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=ol_controller,
                                                  saturate_inputs=True, disturb=True)

        # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
        Qeval = np.diag([10, 10, 1])
        Reval = np.diag([10, 100])
        Seval = np.diag([10, 10, 1])
        print('LQR, Evaluation under closed-loop control')
        evaluate_trajectory(T, x_cl_hist, u_cl_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        print('')
        print('LQR, Evaluation under open-loop control')
        evaluate_trajectory(T, x_ol_hist, u_ol_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        print('')

        # Plotting
        plt.close('all')
        fig1, ax = plot_hist(t_hist, [x_ref_hist, x_ol_hist, x_cl_hist], quantity='state')
        fig2, ax = plot_hist(t_hist, [u_ref_hist, u_ol_hist, u_cl_hist], quantity='input')
        # plot_hist(t_hist, [w_ref_hist, w_ol_hist, w_cl_hist], quantity='disturbance')
        fig3, ax = plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')
        fig4, ax = plot_gain_hist(lqrm_controller.K_hist)

        fig5 = animate(t_hist, x_ref_hist, u_ref_hist, x_ref_hist, u_ref_hist, title='LQR, Open-loop, reference',
                fig_offset=(400, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        fig6 = animate(t_hist, x_cl_hist, u_cl_hist, x_ref_hist, u_ref_hist, title='LQR, Closed-loop, disturbed',
                fig_offset=(1000, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        fig7 = animate(t_hist, x_ol_hist, u_ol_hist, x_ref_hist, u_ref_hist, title='LQR, Open-loop, disturbed',
                fig_offset=(1600, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        if save_plot:
            plot_name = save_file_name + 'lqr_states.png'
            fig1.savefig(plot_name)
            plot_name = save_file_name + 'lqr_inputs.png'
            fig2.savefig(plot_name)
            plot_name = save_file_name + 'lqr_dist_base.png'
            fig3.savefig(plot_name)
            plot_name = save_file_name + 'lqr_gains.png'
            fig4.savefig(plot_name)
            plot_name = save_file_name + 'lqr_ol_ref.png'
            fig5.savefig(plot_name)
            plot_name = save_file_name + 'lqr_cl_dist.png'
            fig6.savefig(plot_name)
            plot_name = save_file_name + 'lqr_ol_dist.png'
            fig7.savefig(plot_name)

    ############# Robust LQR #############
    if run_robust_lqr:
        # Create LQR controller object
        lqrm_controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True)

        # Simulate trajectory with noise and control, forwards in time
        x_cl_hist, u_cl_hist, w_cl_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=lqrm_controller,
                                                  saturate_inputs=True, disturb=True)

        x_ol_hist, u_ol_hist, w_ol_hist = rollout(n, m, T, DT, x0, w_base_hist, controller=ol_controller,
                                                  saturate_inputs=True, disturb=True)

        # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
        Qeval = np.diag([10, 10, 1])
        Reval = np.diag([10, 100])
        Seval = np.diag([10, 10, 1])
        print('Robust LQR, Evaluation under closed-loop control')
        evaluate_trajectory(T, x_cl_hist, u_cl_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        print('')
        print('Robust LQR, Evaluation under open-loop control')
        evaluate_trajectory(T, x_ol_hist, u_ol_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
        print('')

        # Plotting
        plt.close('all')
        fig1, ax = plot_hist(t_hist, [x_ref_hist, x_ol_hist, x_cl_hist], quantity='state')
        fig2, ax = plot_hist(t_hist, [u_ref_hist, u_ol_hist, u_cl_hist], quantity='input')
        # plot_hist(t_hist, [w_ref_hist, w_ol_hist, w_cl_hist], quantity='disturbance')
        fig3, ax = plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')
        fig4, ax = plot_gain_hist(lqrm_controller.K_hist)

        fig5 = animate(t_hist, x_ref_hist, u_ref_hist, x_ref_hist, u_ref_hist, title='Robust LQR, Open-loop, reference',
                fig_offset=(400, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        fig6 = animate(t_hist, x_cl_hist, u_cl_hist, x_ref_hist, u_ref_hist, title='Robust LQR, Closed-loop, disturbed',
                fig_offset=(1000, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        fig7 = animate(t_hist, x_ol_hist, u_ol_hist, x_ref_hist, u_ref_hist, title='Robust LQR, Open-loop, disturbed',
                fig_offset=(1600, 400),
                axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
        if save_plot:
            plot_name = save_file_name + 'lqrm_states.png'
            fig1.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_inputs.png'
            fig2.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_dist_base.png'
            fig3.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_gains.png'
            fig4.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_ol_ref.png'
            fig5.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_cl_dist.png'
            fig6.savefig(plot_name)
            plot_name = save_file_name + 'lqrm_ol_dist.png'
            fig7.savefig(plot_name)

    ############# NMPC #############
    if run_nmpc:
        with_noise = True
        if not with_noise:
            # # solve nmpc problem
            result_data = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=[],
                                              save_plot=False, save_file_name=input_file, drnmpc=False, hnmpc=False)
            # [x_cl_hist_dist, u_cl_hist_dist, collision_flag, crash_idx] = [result_data[i] for i in result_data]
            x_cl_hist = result_data["x_hist"]
            u_cl_hist = result_data["u_hist"]
            collision_flag = result_data["collision_flag"]
            crash_idx = result_data["collision_idx"]

            # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
            Qeval = np.diag([10, 10, 1])
            Reval = np.diag([10, 100])
            Seval = np.diag([10, 10, 1])
            print('NMPC, Evaluation under closed-loop control')
            evaluate_trajectory(T, x_cl_hist, u_cl_hist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
            print('')
            # Plotting
            plt.close('all')
            fig1, ax = plot_hist(t_hist, [x_ref_hist, x_ref_hist, x_cl_hist[0:-1]], quantity='state')
            fig2, ax = plot_hist(t_hist, [u_ref_hist, u_ref_hist, u_cl_hist], quantity='input')
            fig3, ax = plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')

            fig4 = animate(t_hist, x_cl_hist, u_cl_hist, x_ref_hist, u_ref_hist,
                    title='NMPC, Closed-loop, reference',
                    fig_offset=(1000, 400),
                    axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
            if save_plot:
                plot_name = save_file_name + 'nmpc_states_no_dist.png'
                fig1.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_inputs_no_dist.png'
                fig2.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_dist_base_no_dist.png'
                fig3.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_path_no_dist.png'
                fig4.savefig(plot_name)

        # # simulate with disturbance
        # solve nmpc problem
        else:
            result_data  = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=w_base_hist,
                                                               save_plot=False, save_file_name=input_file, drnmpc=True, hnmpc=True)
            # [x_cl_hist_dist, u_cl_hist_dist, collision_flag, crash_idx] = [result_data[i] for i in result_data]
            x_cl_hist_dist = result_data["x_hist"]
            u_cl_hist_dist = result_data["u_hist"]
            collision_flag = result_data["collision_flag"]
            crash_idx = result_data["collision_idx"]
            nlp_failed_flag = result_data['nlp_failed_flag']
            run_time = result_data['run_time']
            print("collision_flag: ", collision_flag)
            print("nlp_failed_flag: ", nlp_failed_flag)

            # Evaluate trajectory in terms of reference tracking, control effort, and disturbance energy
            Qeval = np.diag([10, 10, 1])
            Reval = np.diag([10, 100])
            Seval = np.diag([10, 10, 1])
            print('NMPC, Evaluation under closed-loop control')
            evaluate_trajectory(T, x_cl_hist_dist, u_cl_hist_dist, w_base_hist, x_ref_hist, Qeval, Reval, Seval)
            print('')
            # Plotting
            plt.close('all')
            fig1, ax = plot_hist(t_hist, [x_ref_hist, x_ref_hist, x_cl_hist_dist[0:-1]], quantity='state')
            fig2, ax = plot_hist(t_hist, [u_ref_hist, u_ref_hist, u_cl_hist_dist], quantity='input')
            fig3, ax = plot_hist(t_hist, [w_base_hist, w_base_hist, w_base_hist], quantity='disturbance_base')

            fig4 = animate(t_hist, x_cl_hist_dist, u_cl_hist_dist, x_ref_hist, u_ref_hist,
                    title='NMPC, Closed-loop, disturbed',
                    fig_offset=(1000, 400),
                    axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)
            if save_plot:
                plot_name = save_file_name + 'nmpc_states_dist.png'
                fig1.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_inputs_dist.png'
                fig2.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_dist_base_dist.png'
                fig3.savefig(plot_name)
                plot_name = save_file_name + 'nmpc_path_dist.png'
                fig4.savefig(plot_name)

if __name__ == "__main__":
    run_low_level_controller(run_lqr=False, run_robust_lqr=False, run_nmpc=True)
