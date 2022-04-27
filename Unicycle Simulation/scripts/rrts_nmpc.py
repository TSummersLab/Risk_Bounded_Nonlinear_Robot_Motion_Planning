#!/usr/bin/env python3
"""
Changelog:
New is v1_0:
- Create script

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script runs the RRT*-generated path, extracted by `opt_path.py` with an nmpc low level controller

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

###############################################################################
###############################################################################

# Import all the required libraries
import math
from casadi import *
import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import pickle
from opt_path import load_pickle_file
import os
from plotting import animate

import sys
sys.path.insert(0, '../unicycle')
sys.path.insert(0, '../rrtstar')
from unicycle import lqr, plotting, tracking_controller


import config
STEER_TIME = config.STEER_TIME # Maximum Steering Time Horizon
DT = config.DT # timestep between controls
SAVEPATH = config.SAVEPATH
GOALAREA = config.GOALAREA #[xmin,xmax,ymin,ymax]
VELMAX = config.VELMAX
VELMIN = config.VELMIN
ANGVELMAX = config.ANGVELMAX
ANGVELMIN = config.ANGVELMIN

def sim_state(T, x0, u, f):
    f_value = f(x0, u)
    st = x0+T * f_value.T
    return st

def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = load_pickle_file(input_file)
    ref_states = load_pickle_file(states_file)
    return ref_states, ref_inputs
############################# NMPC FUNCTIONS ##################################

def SetUpSteeringLawParameters(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
    """
    Sets up an IPOPT NLP solver using Casadi SX
    Inputs:
        N: horizon
        T: time step (sec)
        v_max, v_min: maximum and minimum linear velocities in m/s
        omega_max, omega_min: maximum and minimum angular velocities in rad/s
        x_max, x_min, y_max, y_min, theta_max, theta_min: max and min bounds on the states x, y, and theta
    Outputs:
        solver: Casadi NLP solver using ipopt
        f: Casadi continuous time dynamics function
        n_states, n_controls: number of states and controls
        lbx, ubx, lbg, ubg: lower and upper (l,u) state and input (x,g) bounds
    """

    # Define state and input cost matrices
    Q = 1000 * np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.1]])
    R = 1 * np.array([[1.0, 0.0],
                        [0.0, 1.0]])

    # Define symbolic states using Casadi SX
    x = SX.sym('x') # x position
    y = SX.sym('y') # y position
    theta = SX.sym('theta') # heading angle
    states = vertcat(x, y, theta)  # all three states
    n_states = states.size()[0]  # number of symbolic states

    # Define symbolic inputs using Cadadi SX
    v = SX.sym('v') # linear velocity
    omega = SX.sym('omega') # angular velocity
    controls = vertcat(v, omega)  # both controls
    n_controls = controls.size()[0]  # number of symbolic inputs

    # RHS of nonlinear unicycle dynamics (continuous time model)
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)

    # Unicycle continuous time dynamics function
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # Casadi SX trajectory variables/parameters for multiple shooting
    U = SX.sym('U', N, n_controls)  # N trajectory controls
    X = SX.sym('X', N + 1, n_states)  # N+1 trajectory states
    # P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters
    P = SX.sym('P', N + 1, n_states)  # first and last states as independent parameters

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    g = []  # equality constraints
    g.append(X[0, :].T - P[0, :].T)  # add constraint on initial state
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        g.append(X[i + 1, :].T - x_next_.T)
    g.append(X[N, :].T - P[N, :].T)  # constraint on final state

    # Set the nlp problem
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    # Set the nlp problem settings
    opts_setting = {'ipopt.max_iter': 2000,
                    'ipopt.print_level': 0, #4
                    'print_time': 0,
                    'verbose': 0, # 1
                    'error_on_fail': 1}

    # Create a solver that uses IPOPT with above solver settings
    solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # Define the bounds on states and controls
    lbx = []
    ubx = []
    lbg = 0.0
    ubg = 0.0
    # Upper and lower bounds on controls
    for _ in range(N):
        lbx.append(v_min)
        ubx.append(v_max)
    for _ in range(N):
        lbx.append(omega_min)
        ubx.append(omega_max)
    # Upper and lower bounds on states
    for _ in range(N + 1):
        lbx.append(x_min)
        ubx.append(x_max)
    for _ in range(N + 1):
        lbx.append(y_min)
        ubx.append(y_max)
    for _ in range(N + 1):
        lbx.append(theta_min)
        ubx.append(theta_max)

    return solver, f, n_states, n_controls, lbx, ubx, lbg, ubg

def nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx, current_ref_traj, current_ref_inputs):
    """
    Solves the nonlinear steering problem using the solver from SetUpSteeringLawParameters
    Inputs:
        solver: Casadi NLP solver from SetUpSteeringLawParameters
        x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
        n_states, n_controls: number of states and controls
        N: horizon
        T: time step
        lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
        current_ref_traj, current_ref_inputs: reference trajectory and reference inputs as Nx(n_states) ndarrays# TODO: add shapes
    Outputs:
        x_casadi, u_casadi: trajectory states and inputs returned by Casadi
            if solution found:
                states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
                controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
            else, [],[] returned
    """

    # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
    init_states_param = np.linspace(0, 1, N + 1)
    init_states = np.zeros([N + 1, n_states])
    dx = xT - x0
    for i in range(N + 1):
        init_states[i] = (x0 + init_states_param[i] * dx).flatten()

    # Create an initial input trajectory that roughly accomplishes the desired state transfer
    # (using interpolated states to compute rough estimate of controls)
    dist = la.norm(xT[0:2] - x0[0:2])
    ang_dist = xT[2][0] - x0[2][0]
    total_time = N * T
    const_vel = dist / total_time
    const_ang_vel = ang_dist / total_time
    init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)

    ## set parameter
    c_p = np.concatenate((x0, xT))  # start and goal state constraints
    constraint_states = []
    constraint_states.append(x0.reshape(n_states))
    # for _ in range(N):
    #     constraint_states.append(xT.reshape(n_states))
    for ref_state in current_ref_traj:
        constraint_states.append(ref_state.reshape(n_states))
    constraint_states = np.array(constraint_states)
    # c_p = np.concatenate(constraint_states)
    init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    try:
        res = solver(x0=init_decision_vars, p=constraint_states, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    except:
        # raise Exception('NLP failed')
        print('NLP Failed')
        return [], []

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    casadi_result = res['x'].full()

    # Extract the control inputs and states
    u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi

def nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs):
    w = np.zeros([num_steps, num_states])
    return disturbed_nmpc(N, T, v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
                       rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w)

def disturbed_nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w):

    # Set up the Casadi solver
    [solver, f, _, _, lbx, ubx, lbg, ubg] = SetUpSteeringLawParameters(N, T, v_max, v_min,
                                                                       omega_max, omega_min,
                                                                       x_max, x_min,
                                                                       y_max, y_min,
                                                                       theta_max, theta_min)
    # Initialize state and input at goal (which are not part of rrt_states and rrt_inputs
    final_input = [0.0, 0.0]
    final_state = sim_state(T, rrt_states[-1].reshape(3), rrt_inputs[-1], f).full().reshape(3)

    # pad rest of inputs and states with last state and last input for the rest of the horizon (N-1 times)
    rrt_inputs = rrt_inputs.tolist()
    rrt_states = rrt_states.tolist()
    for _ in range(N-1):
        rrt_inputs.append(final_input)
        rrt_states.append(final_state)
    rrt_inputs = np.array(rrt_inputs)
    rrt_states = np.array(rrt_states)

    # nmpc
    nmpc_states = []
    nmpc_ctrls = []
    print('Running nmpc')
    for itr in range(num_steps-1):
        # print(itr+1, '/', num_steps-1)

        # get reference trajectory and inputs based on the RRT trajectory
        current_ref_traj = rrt_states[itr+1:itr+N+1]
        current_ref_inputs = rrt_inputs[itr+1:itr+N+1]

        # get current state and current goal state
        if itr == 0: # first iteration
            current_state = rrt_states[0].reshape(num_states, 1)
            nmpc_states.append(current_state.reshape(num_states, 1))
        else: # for future iterations, start at last nmpc state reached
            current_state = nmpc_states[-1].reshape(num_states, 1)
        current_goal_state = rrt_states[itr+N].reshape(num_states, 1) # goal state comes from RRT* plan

        x_casadi, u_casadi = nonlinsteer(solver, current_state, current_goal_state, num_states, num_inputs,
                                         N, T, lbg, lbx, ubg, ubx, current_ref_traj, current_ref_inputs)
        if x_casadi == []:
            print("nmpc failed at itr: ", itr)
            return [], []

        nmpc_input = u_casadi[0] # input to apply

        # use this and simulate with f one step then repeat
        next_state = sim_state(T, current_state.reshape(num_states), nmpc_input, f).full().reshape(num_states)

        # update the state and input
        nmpc_states.append(next_state.reshape(num_states,1) + w[itr].reshape(num_states,1)) #w[itr].reshape(num_states) # npr.normal(0,0.01,3) #
        nmpc_ctrls.append(nmpc_input.reshape(num_inputs,1))

    print('Done with nmpc')
    # # add a final while loop to keep robot at center of goal region
    # xGoal_center = GOALAREA[0] + (GOALAREA[1] - GOALAREA[0]) / 2.0
    # yGoal_center = GOALAREA[2] + (GOALAREA[3] - GOALAREA[2]) / 2.0
    # goal_center = [xGoal_center, yGoal_center, rrt_states[-1][2]]
    # final_input = [0.0, 0.0]
    # current_state = rrt_states[-1].reshape(3)
    nmpc_states = np.array(nmpc_states).reshape(len(nmpc_states),num_states)
    nmpc_ctrls = np.array(nmpc_ctrls).reshape(len(nmpc_ctrls), num_inputs)
    distance_error = la.norm(final_state[0:2] - nmpc_states[-1][0:2])
    print('Final error away from RRT* goal:', distance_error)

    return nmpc_states, nmpc_ctrls

###############################################################################
####################### FUNCTION CALLED BY MAIN() #############################
###############################################################################

#TODO:change this to support different min and max values
def rrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max, omega_max, x_max, y_max, theta_max, w=[],
                      animate_results = False, save_plot = False,  save_file_name = "",
                      ax_lim = [-5, 5, -5, 5], robot_w = 0.2, robot_h = 0.5, wheel_w = 0.5, wheel_h = 0.005):
    """
    runs an nmpc low level controller for the rrt* path
    Inputs:
        input_file: file name (only) for the optimal inputs
        file_path: path to input_file
        v_max, omega_max, x_max, y_max, theta_max: maximum linear and angular velocity and maximum x,y,theta values
        w: generated disturbance
        animate_results: True --> animate, False --> don't animate
        save_plot: True --> save plot, False --> don't save plot
        ax_lim: axis limits for animation
        robot_w: robot width for animation
        robot_h: robot height for animation
        wheel_w: robot wheel width for animation
        wheel_h: robot wheel height for animation
    """

    # load inputs and states
    rrt_states = x_ref_hist
    rrt_inputs = u_ref_hist

    # extract reference states and inputs from the rrt plan
    x_ref_hist = np.array(rrt_states).reshape(num_steps, n)
    u_ref_hist = np.array(rrt_inputs).reshape(num_steps, m)


    # time vector
    t0, tf = 0, (num_steps) * DT
    # Sampling period
    Ts = DT
    # Time history
    t_hist = np.arange(t0, tf, Ts)


    # extract the x and y states in the rrt plan
    x_orig = np.array(rrt_states).reshape(num_steps, n)[:, 0]
    y_orig = np.array(rrt_states).reshape(num_steps, n)[:, 1]

    if w == []: # no disturbance
        # nmpc with no disturbance
        all_states_cl, all_inputs_cl = nmpc(nmpc_horizon, DT, v_max, -v_max, omega_max, -omega_max, x_max, -x_max, y_max,
                                            -y_max, theta_max, -theta_max,
                                            rrt_states, rrt_inputs, GOALAREA, num_steps, n, m)
        # if it failed, stop
        if all_states_cl == []:
            print('Failed')
            return

        # get the x,y states of nmpc with no disturbances
        x_cl = np.array(all_states_cl)[:, 0]
        y_cl = np.array(all_states_cl)[:, 1]
        # plot sampled points, selected samples for optimal trajectory, and optimal trajectory
        ax = plt.axes()
        plt.plot(x_orig, y_orig, 'o', color='black')
        plt.plot(x_cl, y_cl, 'x', color='red')
        if save_plot:
            plot_name = save_file_name + '_plot_nmpc.png'
            plt.savefig(plot_name)
        plt.show()

        return all_states_cl, all_inputs_cl

    else:
        # run nmpc with disturbance
        all_states_cl_dist, all_inputs_cl_dist = disturbed_nmpc(nmpc_horizon, DT, v_max, -v_max, omega_max, -omega_max,
                                                                x_max, -x_max, y_max, -y_max, theta_max, -theta_max,
                                                                rrt_states, rrt_inputs, GOALAREA, num_steps, n, m, w)

        if all_states_cl_dist == []:
            print('Failed')
            return

        x_cl_dist = np.array(all_states_cl_dist)[:, 0]
        y_cl_dist = np.array(all_states_cl_dist)[:, 1]

        # plot sampled points, selected samples for optimal trajectory, and optimal trajectory
        ax = plt.axes()
        plt.plot(x_orig, y_orig, 'o', color='black')
        plt.plot(x_cl_dist, y_cl_dist, 'x', color='red')
        if save_plot:
            plot_name = save_file_name + '_plot_nmpc.png'
            plt.savefig(plot_name)
        plt.show()

        return all_states_cl_dist, all_inputs_cl_dist


if __name__ == '__main__':
    # load file
    input_file = "OptTraj_short_v1_0_1607441105_inputs"
    x_ref_hist, u_ref_hist = load_ref_traj(input_file)
    rrt_states = x_ref_hist
    rrt_inputs = u_ref_hist

    v_max = VELMAX  # maximum linear velocity (m/s)
    omega_max = ANGVELMAX # 0.125 * (2 * np.pi)  # maximum angular velocity (rad/s)
    x_max = 5  # maximum state in the horizontal direction
    y_max = 5  # maximum state in the vertical direction
    theta_max = np.inf  # maximum state in the theta direction
    ax_lim = [-6, 6, -6, 6]
    robot_w = 0.2 / 2
    robot_h = 0.5 / 2
    wheel_w = 0.5 / 2
    wheel_h = 0.005 / 2
    nmpc_horizon = STEER_TIME

    # Number of states, inputs
    _, n = rrt_states.shape
    num_steps, m = rrt_inputs.shape

    # generate disturbance
    # Time start, end
    t0, tf = 0, (num_steps) * DT
    # Sampling period
    Ts = DT
    # Time history
    t_hist = np.arange(t0, tf, Ts)
    # Number of time steps
    T = t_hist.size
    # Initial state and disturbance
    x0 = np.array(rrt_states[0, :])
    # Generate base disturbance sequence
    w_base_hist = tracking_controller.generate_disturbance_hist(T, Ts, scale=1)
    plt.plot(t_hist, w_base_hist[:, 0])
    plt.show()
    plt.plot(t_hist, w_base_hist[:, 1])
    plt.show()
    plt.plot(t_hist, w_base_hist[:, 2])
    plt.show()

    # simulate with no disturbances
    all_states_cl, all_inputs_cl = rrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max, omega_max, x_max,
                                                     y_max, theta_max, w=[],
                                                     animate_results=True, save_plot=False, save_file_name=input_file,
                                                     ax_lim=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w,
                                                     wheel_h=wheel_h)
    animate(t_hist, all_states_cl, all_inputs_cl, x_ref_hist, u_ref_hist,
            title='NMPC, Closed-loop, reference',
            fig_offset=(1000, 400),
            axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)


    # simulate with disturbance
    all_states_cl_dist, all_inputs_cl_dist = rrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max,
                                                               omega_max, x_max, y_max, theta_max, w=w_base_hist,
                                                               animate_results=True, save_plot=False,
                                                               save_file_name=input_file,
                                                               ax_lim=ax_lim, robot_w=robot_w, robot_h=robot_h,
                                                               wheel_w=wheel_w, wheel_h=wheel_h)

    animate(t_hist, all_states_cl_dist, all_inputs_cl_dist, x_ref_hist, u_ref_hist,
            title='NMPC, Closed-loop, referdisturbedence',
            fig_offset=(1000, 400),
            axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)