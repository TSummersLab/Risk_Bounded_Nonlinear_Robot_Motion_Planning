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
import copy

import sys
sys.path.insert(0, '../unicycle')
sys.path.insert(0, '../rrtstar')
from unicycle import lqr, plotting, tracking_controller


import config
STEER_TIME = config.STEER_TIME # Maximum Steering Time Horizon
DT = config.DT # timestep between controls
SAVEPATH = config.SAVEPATH
GOALAREA = config.GOALAREA #[xmin,xmax,ymin,ymax]
RANDAREA = copy.copy(config.RANDAREA) # [xmin,xmax,ymin,ymax]
VELMAX = config.VELMAX
VELMIN = config.VELMIN
ANGVELMAX = config.ANGVELMAX
ANGVELMIN = config.ANGVELMIN
ROBRAD = config.ROBRAD  # radius of robot (added as padding to environment bounds and the obstacles
OBSTACLELIST = copy.copy(config.OBSTACLELIST)  # [ox,oy,wd,ht]

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

# def SetUpSteeringLawParameters(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
#     """
#     Sets up an IPOPT NLP solver using Casadi Opti
#     Inputs:
#         N: horizon
#         T: time step (sec)
#         v_max, v_min: maximum and minimum linear velocities in m/s
#         omega_max, omega_min: maximum and minimum angular velocities in rad/s
#         x_max, x_min, y_max, y_min, theta_max, theta_min: max and min bounds on the states x, y, and theta
#     Outputs:
#         solver: Casadi NLP solver using ipopt
#         f: Casadi continuous time dynamics function
#         n_states, n_controls: number of states and controls
#         lbx, ubx, lbg, ubg: lower and upper (l,u) state and input (x,g) bounds
#     """
#
#     # Define state and input cost matrices
#     Q = 1000 * np.array([[1.0, 0.0, 0.0],
#                   [0.0, 1.0, 0.0],
#                   [0.0, 0.0, 0.1]])
#     R = 1 * np.array([[1.0, 0.0],
#                         [0.0, 1.0]])
#     QT = 1000 * Q
#
#     opti = casadi.Opti()
#
#     # Define symbolic states using Casadi Opti
#     # x = SX.sym('x') # x position
#     x = opti.variable()
#     # y = SX.sym('y') # y position
#     y = opti.variable()
#     # theta = SX.sym('theta') # heading angle
#     theta = opti.variable()
#     states = vertcat(x, y, theta)  # all three states
#     n_states = states.size()[0]  # number of symbolic states
#
#     # Define symbolic inputs using Cadadi SX
#     # v = SX.sym('v') # linear velocity
#     v = opti.variable()
#     # omega = SX.sym('omega') # angular velocity
#     omega = opti.variable()
#     controls = vertcat(v, omega)  # both controls
#     n_controls = controls.size()[0]  # number of symbolic inputs
#
#     # RHS of nonlinear unicycle dynamics (continuous time model)
#     rhs = horzcat(v * cos(theta), v * sin(theta), omega)
#
#     # Unicycle continuous time dynamics function
#     f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
#
#     # Casadi SX trajectory variables/parameters for multiple shooting
#     # U = SX.sym('U', N, n_controls)  # N trajectory controls
#     U = opti.variable(N, n_controls)
#     # X = SX.sym('X', N + 1, n_states)  # N+1 trajectory states
#     X = opti.variable(N+1, n_states)
#     # P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters
#     # P = SX.sym('P', N + 1, n_states)  # all N+1 states as independent parameters
#     P = opti.parameter(N+1, n_states)
#
#     discrete = [False]*(N*n_controls + (N+1)*n_states)
#
#     # Concatinate the decision variables (inputs and states)
#     opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))
#
#     # Cost function
#     obj = 0  # objective/cost
#     # g = []  # equality constraints
#     # g.append(X[0, :].T - P[0, :].T)  # add constraint on initial state
#     opti.subject_to(X[0, :].T == P[0, :].T)
#     for i in range(N):
#         # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
#         # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
#         obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
#         obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])
#
#         # compute the next state from the dynamics
#         x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]
#
#         # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
#         # g.append(X[i + 1, :].T - x_next_.T)
#         opti.subject_to(X[i + 1, :].T == x_next_.T)
#     obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T]) # TODO: Either keep this only without the equality constraint
#     opti.minimize(obj)
#     # g.append(X[N, :].T - P[N, :].T)  # constraint on final state # TODO: or replace this by |X - P| < tolerance constraint
#     # opti.subject_to(X[N, :].T == P[N, :].T) # TODO: uncomment ??? Maybe???
#
#     # state environment constraints
#     opti.subject_to(opti.bounded(x_min, X[:,0], x_max))
#     opti.subject_to(opti.bounded(y_min, X[:,1], y_max))
#     opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
#     # input constraints
#     opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
#     opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))
#
#
#     # # obstacle constraints TODO: add obstacle constraints
#     # obs_edges, _ = get_padded_edges()
#     # num_obs = len(obs_edges)
#     # eps = 0.0001 # some small tolerance
#     # delta = opti.variable(4*num_obs)
#     # opti.subject_to(opti.bounded(0-eps, delta, 1+eps))
#     # discrete += [True] * (4 * num_obs)
#     # # for obs_num, obs in enumerate(obs_edges):
#     # #     top = obs["top"]
#     # #     bottom = obs["bottom"]
#     # #     right = obs["right"]
#     # #     left = obs["left"]
#     # #     Ml = x_max - left + 1
#     # #     ml = x_min - left - 1
#     # #     Mr = x_max - right + 1
#     # #     mr = x_min - right - 1
#     # #     Mb = y_max - bottom - 1
#     # #     mb = y_min - bottom - 1
#     # #     Mt = y_max - top + 1
#     # #     mt = y_min - top - 1
#     # #
#     # #     opti.subject_to(X[:, 0] + Ml * delta[4 * obs_num + 0] <= Ml + left)
#     # #     opti.subject_to(X[:, 0] - (ml - eps) * delta[4 * obs_num + 0] >= left + eps)
#     # #
#     # #     opti.subject_to(X[:, 0] + mr * delta[4 * obs_num + 1] >= mr + right)
#     # #     opti.subject_to(X[:, 0] - (Mr + eps) * delta[4 * obs_num + 1] <= right - eps)
#     # #
#     # #     opti.subject_to(X[:, 1] + Mb * delta[4 * obs_num + 2] <= Mb + bottom)
#     # #     opti.subject_to(X[:, 1] - (mb - eps) * delta[4 * obs_num + 2] >= bottom + eps)
#     # #
#     # #     opti.subject_to(X[:, 1] + mt * delta[4 * obs_num + 3] >= mt + top)
#     # #     opti.subject_to(X[:, 1] - (Mt + eps) * delta[4 * obs_num + 3] <= top - eps)
#     # #
#     # #     opti.subject_to(1-eps <= delta[4 * obs_num + 0] + delta[4 * obs_num + 1] + delta[4 * obs_num + 2] + delta[4 * obs_num + 3])
#
#     # obstacle constraints using Big-M formulation TODO: TRY THE CONVEX-HULL REFORMULATION https://optimization.mccormick.northwestern.edu/index.php/Disjunctive_inequalities (it might be faster)
#     # TODO: THIS BIG-M FORMULATION WORKS !!!!!!!!!!!!!!!!! YAY
#     obs_edges, env_edges = get_padded_edges()
#     x_max_env = env_edges["right"]
#     x_min_env = env_edges["left"]
#     y_max_env = env_edges["top"]
#     y_min_env = env_edges["bottom"]
#
#     num_obs = len(obs_edges)
#     eps = 0.0001  # some small tolerance
#     delta = opti.variable(4 * num_obs)
#     opti.subject_to(opti.bounded(0 - eps, delta, 1 + eps))
#     discrete += [True] * (4 * num_obs)
#     M = 10 # a large upper bound on x and y
#     for obs_num, obs in enumerate(obs_edges):
#         top = obs["top"]
#         bottom = obs["bottom"]
#         right = obs["right"]
#         left = obs["left"]
#
#         opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 0]) + right, X[:, 0], x_max_env + M * (1 - delta[4 * obs_num + 0])))
#         opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 1]) + x_min_env, X[:, 0], left + M * (1 - delta[4 * obs_num + 1])))
#         opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 2]) + top, X[:, 1], y_max_env + M * (1 - delta[4 * obs_num + 2])))
#         opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 3]) + y_min_env, X[:, 1], bottom + M * (1 - delta[4 * obs_num + 3])))
#
#         opti.subject_to(
#             1 <= delta[4 * obs_num + 0] + delta[4 * obs_num + 1] + delta[4 * obs_num + 2] + delta[4 * obs_num + 3])
#
#     # Set the nlp problem settings
#     opts_setting = {'ipopt.max_iter': 2000,
#                     'ipopt.print_level': 4,  # 4
#                     'print_time': 0,
#                     'verbose': 1,  # 1
#                     'ipopt.acceptable_tol':1e-8,
#                     'ipopt.acceptable_obj_change_tol':1e-6,
#                     'error_on_fail': 1}
#     args = dict(discrete=discrete) #dict(discrete=[False, False, True])
#
#     # Set the nlp problem
#     # nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}
#
#     # opti.solver('ipopt', opts_setting)
#     opti.solver("bonmin", args)
#     solver = opti
#
#     # # Create a solver that uses IPOPT with above solver settings
#     # solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
#
#     # # Define the bounds on states and controls
#     # lbx = []
#     # ubx = []
#     # lbg = 0.0
#     # ubg = 0.0
#     # # Upper and lower bounds on controls
#     # for _ in range(N):
#     #     lbx.append(v_min)
#     #     ubx.append(v_max)
#     # for _ in range(N):
#     #     lbx.append(omega_min)
#     #     ubx.append(omega_max)
#     # # Upper and lower bounds on states
#     # for _ in range(N + 1):
#     #     lbx.append(x_min)
#     #     ubx.append(x_max)
#     # for _ in range(N + 1):
#     #     lbx.append(y_min)
#     #     ubx.append(y_max)
#     # for _ in range(N + 1):
#     #     lbx.append(theta_min)
#     #     ubx.append(theta_max)
#
#     return solver, f, n_states, n_controls, U, X, P, delta
#
#
# def SetUpSteeringLawParameters2(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
#     """
#     Sets up an IPOPT NLP solver using Casadi Opti
#     Inputs:
#         N: horizon
#         T: time step (sec)
#         v_max, v_min: maximum and minimum linear velocities in m/s
#         omega_max, omega_min: maximum and minimum angular velocities in rad/s
#         x_max, x_min, y_max, y_min, theta_max, theta_min: max and min bounds on the states x, y, and theta
#     Outputs:
#         solver: Casadi NLP solver using ipopt
#         f: Casadi continuous time dynamics function
#         n_states, n_controls: number of states and controls
#         lbx, ubx, lbg, ubg: lower and upper (l,u) state and input (x,g) bounds
#     """
#
#     # Define state and input cost matrices
#     Q = 1000 * np.array([[1.0, 0.0, 0.0],
#                   [0.0, 1.0, 0.0],
#                   [0.0, 0.0, 0.1]])
#     R = 1 * np.array([[1.0, 0.0],
#                         [0.0, 1.0]])
#     QT = 1000 * Q
#
#     opti = casadi.Opti()
#
#     # Define symbolic states using Casadi Opti
#     # x = SX.sym('x') # x position
#     x = opti.variable()
#     # y = SX.sym('y') # y position
#     y = opti.variable()
#     # theta = SX.sym('theta') # heading angle
#     theta = opti.variable()
#     states = vertcat(x, y, theta)  # all three states
#     n_states = states.size()[0]  # number of symbolic states
#
#     # Define symbolic inputs using Cadadi SX
#     # v = SX.sym('v') # linear velocity
#     v = opti.variable()
#     # omega = SX.sym('omega') # angular velocity
#     omega = opti.variable()
#     controls = vertcat(v, omega)  # both controls
#     n_controls = controls.size()[0]  # number of symbolic inputs
#
#     # RHS of nonlinear unicycle dynamics (continuous time model)
#     rhs = horzcat(v * cos(theta), v * sin(theta), omega)
#
#     # Unicycle continuous time dynamics function
#     f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
#
#     # Casadi SX trajectory variables/parameters for multiple shooting
#     # U = SX.sym('U', N, n_controls)  # N trajectory controls
#     U = opti.variable(N, n_controls)
#     # X = SX.sym('X', N + 1, n_states)  # N+1 trajectory states
#     X = opti.variable(N+1, n_states)
#     # P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters
#     # P = SX.sym('P', N + 1, n_states)  # all N+1 states as independent parameters
#     P = opti.parameter(N+1, n_states)
#
#     # parameters for x,y state upper and lower bounds
#     XMAX = opti.parameter(N + 1, 1)
#     XMIN = opti.parameter(N + 1, 1)
#     YMAX = opti.parameter(N + 1, 1)
#     YMIN = opti.parameter(N + 1, 1)
#
#     discrete = [False]*(N*n_controls + (N+1)*n_states)
#
#     # Concatinate the decision variables (inputs and states)
#     opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))
#
#     # Cost function
#     obj = 0  # objective/cost
#     # g = []  # equality constraints
#     # g.append(X[0, :].T - P[0, :].T)  # add constraint on initial state
#     opti.subject_to(X[0, :].T == P[0, :].T)
#     for i in range(N):
#         # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
#         # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
#         obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
#         obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])
#
#         # compute the next state from the dynamics
#         x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]
#
#         # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
#         # g.append(X[i + 1, :].T - x_next_.T)
#         opti.subject_to(X[i + 1, :].T == x_next_.T)
#     obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T]) # TODO: Either keep this only without the equality constraint
#     opti.minimize(obj)
#     # g.append(X[N, :].T - P[N, :].T)  # constraint on final state # TODO: or replace this by |X - P| < tolerance constraint
#     # opti.subject_to(X[N, :].T == P[N, :].T) # TODO: uncomment ??? Maybe???
#
#     # state environment constraints
#     opti.subject_to(opti.bounded(XMIN, X[:,0], XMAX))
#     opti.subject_to(opti.bounded(YMIN, X[:,1], YMAX))
#     opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
#     # input constraints
#     opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
#     opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))
#
#
#     # obstacle constraints TODO: add obstacle constraints
#     obs_edges, _ = get_padded_edges()
#     num_obs = len(obs_edges)
#     eps = 0.0001 # some small tolerance
#     delta = opti.variable(4*num_obs)
#     opti.subject_to(opti.bounded(0-eps, delta, 1+eps))
#     discrete += [True] * (4 * num_obs)
#
#     # Set the nlp problem settings
#     opts_setting = {'ipopt.max_iter': 2000,
#                     'ipopt.print_level': 0,  # 4
#                     'print_time': 0,
#                     'verbose': 0,  # 1
#                     'ipopt.acceptable_tol':1e-8,
#                     'ipopt.acceptable_obj_change_tol':1e-6,
#                     'error_on_fail': 1}
#     args = dict(discrete=discrete) #dict(discrete=[False, False, True])
#
#     # Set the nlp problem
#     # nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}
#
#     # opti.solver('ipopt', opts_setting)
#     opti.solver("bonmin", args)
#     solver = opti
#
#     return solver, f, n_states, n_controls, U, X, P, delta, XMIN, XMAX, YMIN, YMAX
#

def SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min):
    """
    Sets up an IPOPT NLP solver using Casadi Opti
    Inputs:
        N: horizon
        T: time step (sec)
        v_max, v_min: maximum and minimum linear velocities in m/s
        omega_max, omega_min: maximum and minimum angular velocities in rad/s
    Outputs:
    solver, f, n_states, n_controls, U, X, P, delta
        solver: Casadi NLP solver using bonmin
        f: Casadi continuous time dynamics function
        n_states, n_controls: number of states and controls
        U, X: Casadi input and state variables (N x n_controls and (N+1)x n_states matrices)
        P: Casadi desired state parameters ((N+1) x n_states matrix)
        Delta: Casadi 0-1 variables for constraints (4*num_obs vector)
    """

    # Define state and input cost matrices
    Q = 1000 * np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.1]])
    R = 1 * np.array([[1.0, 0.0],
                        [0.0, 1.0]])
    QT = 1000 * Q


    opti = casadi.Opti()

    # Define symbolic states using Casadi Opti
    x = opti.variable()
    y = opti.variable()
    theta = opti.variable()
    states = vertcat(x, y, theta)  # all three states
    n_states = states.size()[0]  # number of symbolic states

    # Define symbolic inputs using Cadadi SX
    v = opti.variable()
    omega = opti.variable()
    controls = vertcat(v, omega)  # both controls
    n_controls = controls.size()[0]  # number of symbolic inputs

    # RHS of nonlinear unicycle dynamics (continuous time model)
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)

    # Unicycle continuous time dynamics function
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # Casadi Opti trajectory variables/parameters for multiple shooting
    U = opti.variable(N, n_controls)
    X = opti.variable(N+1, n_states)
    P = opti.parameter(N+1, n_states)
    discrete = [False]*(N*n_controls + (N+1)*n_states) # specify U and X to be continuous variables

    # Cost function
    obj = 0  # objective/cost
    opti.subject_to(X[0, :].T == P[0, :].T)
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T]) # quadratic penalty on deviation from reference state

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
        opti.subject_to(X[i + 1, :].T == x_next_.T)

    # we might not be able to get back to the original target goal state
    # alternatively, we have a large penalty of being away from it
    obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T])

    # minimize this objective
    opti.minimize(obj)

    # state environment constraints
    _, env_edges = get_padded_edges()
    x_max_env = env_edges["right"]
    x_min_env = env_edges["left"]
    y_max_env = env_edges["top"]
    y_min_env = env_edges["bottom"]
    opti.subject_to(opti.bounded(x_min_env, X[:, 0], x_max_env))
    opti.subject_to(opti.bounded(y_min_env, X[:, 1], y_max_env))
    opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
    # input constraints
    opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
    opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))

    # create a dict of the discrete flags
    args = dict(discrete=discrete)
    # specify the solver
    opti.solver("bonmin", args)

    solver = opti # solver instance to return

    DELTA = []
    STARTIDX = []
    return solver, f, n_states, n_controls, U, X, P, DELTA, STARTIDX

def SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min):
    """
    Sets up a BONMIN MINLP solver using Casadi Opti
    Collision avoidance is encoded with Big-M formulation

    Inputs:
        N: horizon
        T: time step (sec)
        v_max, v_min: maximum and minimum linear velocities in m/s
        omega_max, omega_min: maximum and minimum angular velocities in rad/s
    Outputs:
    solver, f, n_states, n_controls, U, X, P, delta
        solver: Casadi NLP solver using bonmin
        f: Casadi continuous time dynamics function
        n_states, n_controls: number of states and controls
        U, X: Casadi input and state variables (N x n_controls and (N+1)x n_states matrices)
        P: Casadi desired state parameters ((N+1) x n_states matrix)
        Delta: Casadi 0-1 variables for constraints (4*num_obs vector)
    """

    # Define state and input cost matrices
    Q = 1000 * np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.1]])
    R = 1 * np.array([[1.0, 0.0],
                        [0.0, 1.0]])
    QT = 1000 * Q


    opti = casadi.Opti()

    # Define symbolic states using Casadi Opti
    x = opti.variable()
    y = opti.variable()
    theta = opti.variable()
    states = vertcat(x, y, theta)  # all three states
    n_states = states.size()[0]  # number of symbolic states

    # Define symbolic inputs using Cadadi SX
    v = opti.variable()
    omega = opti.variable()
    controls = vertcat(v, omega)  # both controls
    n_controls = controls.size()[0]  # number of symbolic inputs

    # RHS of nonlinear unicycle dynamics (continuous time model)
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)

    # Unicycle continuous time dynamics function
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # Casadi Opti trajectory variables/parameters for multiple shooting
    U = opti.variable(N, n_controls)
    X = opti.variable(N+1, n_states)
    P = opti.parameter(N+1, n_states)
    discrete = [False]*(N*n_controls + (N+1)*n_states) # specify U and X to be continuous variables

    # Cost function
    obj = 0  # objective/cost
    opti.subject_to(X[0, :].T == P[0, :].T)
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T]) # quadratic penalty on deviation from reference state

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
        opti.subject_to(X[i + 1, :].T == x_next_.T)

    # we might not be able to get back to the original target goal state
    # alternatively, we have a large penalty of being away from it
    obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T])

    # minimize this objective
    opti.minimize(obj)

    # state environment constraints
    opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf)) # theta only now (x,y states added later)
    # input constraints
    opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
    opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))


    # obstacle constraints using Big-M formulation TODO: TRY THE CONVEX-HULL REFORMULATION https://optimization.mccormick.northwestern.edu/index.php/Disjunctive_inequalities (it might be faster)
    obs_edges, env_edges = get_padded_edges()
    x_max_env = env_edges["right"]
    x_min_env = env_edges["left"]
    y_max_env = env_edges["top"]
    y_min_env = env_edges["bottom"]

    opti.subject_to(opti.bounded(x_min_env, X[:, 0], x_max_env))
    opti.subject_to(opti.bounded(y_min_env, X[:, 1], y_max_env))

    num_obs = len(obs_edges)
    DELTA = opti.variable(4 * num_obs) # 0-1 variables to indicate if an obstacle is hit
    opti.subject_to(opti.bounded(0, DELTA, 1))
    discrete += [True] * (4 * num_obs) # specify the delta variables to be discrete (with above bound --> 0-1 variables)
    M = max(x_max_env-x_min_env, y_max_env-y_min_env) + 1 # 10 # a large upper bound on x and y
    STARTIDX = opti.parameter(1) # specify which points in the horizon should have collision avoidance enforced
    for obs_num, obs in enumerate(obs_edges):
        # for every obstacle
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        # add Big-M formulation disjunctive constraints
        opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 0]) + right, X[1:, 0], x_max_env + M * (1 - DELTA[4 * obs_num + 0])))  # be to the right of the obstacle
        opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 1]) + x_min_env, X[1:, 0], left + M * (1 - DELTA[4 * obs_num + 1])))  # be to the left of the obstacle
        opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 2]) + top, X[1:, 1], y_max_env + M * (1 - DELTA[4 * obs_num + 2])))  # be to the top of the obstacle
        opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 3]) + y_min_env, X[1:, 1], bottom + M * (1 - DELTA[4 * obs_num + 3])))  # be to the bottom of the obstacle

        # require at least one of these constraints to be true
        opti.subject_to(
            1 <= DELTA[4 * obs_num + 0] + DELTA[4 * obs_num + 1] + DELTA[4 * obs_num + 2] + DELTA[4 * obs_num + 3])

    # create a dict of the discrete flags
    args = dict(discrete=discrete)
    # specify the solver
    opti.solver("bonmin", args)

    solver = opti # solver instance to return

    return solver, f, n_states, n_controls, U, X, P, DELTA, STARTIDX

def nonlinsteerNoColAvoid(solver, x0, xT, n_states, n_controls, N, T, U, X, P, DELTA, STARTIDX, current_ref_traj, current_ref_inputs, start_idx):
    """
    Solves the nonlinear steering problem using the solver from SetUpSteeringLawParametersBigM
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
    constraint_states = []
    constraint_states.append(x0.reshape(n_states))

    for ref_state in current_ref_traj:
        constraint_states.append(ref_state.reshape(n_states))

    init_inputs = []
    for ref_input in current_ref_inputs:
        init_inputs.append(ref_input.reshape(n_controls))
    init_inputs = np.array(init_inputs)

    constraint_states = np.array(constraint_states)
    solver.set_value(P, constraint_states)
    solver.set_initial(X, constraint_states)
    solver.set_initial(U, init_inputs)
    # solver.set_initial(X, init_states)
    # solver.set_initial(U, init_inputs)
    try:
        res = solver.solve()
    except:
        print('Steering NLP Failed')
        return [], []

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U) # shape: (N, n_controls)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X) # shape: # (N+1, n_states)

    return x_casadi, u_casadi

def nonlinsteerBigM(solver, x0, xT, n_states, n_controls, N, T, U, X, P, DELTA, STARTIDX, current_ref_traj, current_ref_inputs, start_idx):
    """
    Solves the nonlinear steering problem using the solver from SetUpSteeringLawParametersBigM
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
    constraint_states = []
    constraint_states.append(x0.reshape(n_states))


    for ref_state in current_ref_traj:
        constraint_states.append(ref_state.reshape(n_states))
    constraint_states = np.array(constraint_states)

    init_inputs = []
    for ref_input in current_ref_inputs:
        init_inputs.append(ref_input.reshape(n_controls))
    init_inputs = np.array(init_inputs)

    solver.set_value(P, constraint_states)
    solver.set_value(STARTIDX, start_idx)
    solver.set_initial(X, constraint_states)
    solver.set_initial(U, init_inputs)
    try:
        res = solver.solve()
    except:
        print('Steering NLP Failed')
        return [], []

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U) # shape: (N, n_controls)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X) # shape: # (N+1, n_states)

    print('delta', res.value(DELTA))

    return x_casadi, u_casadi

def nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs):
    w = np.zeros([num_steps, num_states])
    return disturbed_nmpc(N, T, v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
                       rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w)


# def disturbed_nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
#          rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w):
#
#     col_avoid = False
#
#     # TODO: remove x_min, x_max, y_min, y_max from inputs
#     _, env_edges = get_padded_edges()
#     x_max = env_edges["right"]
#     x_min = env_edges["left"]
#     y_max = env_edges["top"]
#     y_min = env_edges["bottom"]
#     # Set up the Casadi solver
#     if col_avoid:
#         [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)
#     else:
#         [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)
#
#     # # Fast solver without collision avoidance
#     # [solverF, _, _, _, UF, XF, PF, _] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)
#     # # Slower solver with collision avoidance
#     # [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)
#
#     final_input = [0.0, 0.0] # final input
#     final_state = sim_state(T, rrt_states[-1].reshape(3), rrt_inputs[-1], f).full().reshape(3) # final state
#
#     # pad rest of inputs and states with last state and last input for the rest of the horizon (N-1 times)
#     rrt_inputs = rrt_inputs.tolist()
#     rrt_states = rrt_states.tolist()
#     for _ in range(N-1): #(N-1)
#         rrt_inputs.append(final_input)
#         rrt_states.append(final_state)
#     rrt_inputs = np.array(rrt_inputs)
#     rrt_states = np.array(rrt_states)
#
#     # nmpc
#     nmpc_states = []
#     nmpc_ctrls = []
#     print('Running nmpc')
#     for itr in range(num_steps-1):
#         # print(itr+1, '/', num_steps-1)
#
#         # get reference trajectory and inputs based on the RRT trajectory (excludes current state)
#         current_ref_traj = rrt_states[itr+1:itr+N+1]
#         current_ref_inputs = rrt_inputs[itr+1:itr+N+1]
#
#         # get current state and current goal state
#         if itr == 0: # first iteration
#             current_state = rrt_states[0].reshape(num_states, 1)
#             nmpc_states.append(current_state.reshape(num_states, 1))
#         else: # for future iterations, start at last nmpc state reached
#             current_state = nmpc_states[-1].reshape(num_states, 1)
#         current_goal_state = rrt_states[itr+N].reshape(num_states, 1) # goal state comes from RRT* plan
#
#         if col_avoid:
#             x_casadi, u_casadi = nonlinsteerBigM(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
#                                                  X, P, DELTA, current_ref_traj, current_ref_inputs)
#         else:
#             x_casadi, u_casadi = nonlinsteerNoColAvoid(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
#                                                  X, P, DELTA, current_ref_traj, current_ref_inputs)
#
#         if x_casadi == []:
#             print("nmpc failed at itr: ", itr)
#             break
#             # return [], []
#
#         nmpc_input = u_casadi[0] # input to apply
#
#         # use this and simulate with f one step then repeat
#         next_state = sim_state(T, current_state.reshape(num_states), nmpc_input, f).full().reshape(num_states)
#
#         # update the state and input
#         nmpc_states.append(next_state.reshape(num_states,1) + w[itr].reshape(num_states,1)) #w[itr].reshape(num_states) # npr.normal(0,0.01,3) #
#         nmpc_ctrls.append(nmpc_input.reshape(num_inputs,1))
#
#     print('Done with nmpc')
#     # # add a final while loop to keep robot at center of goal region
#     # xGoal_center = GOALAREA[0] + (GOALAREA[1] - GOALAREA[0]) / 2.0
#     # yGoal_center = GOALAREA[2] + (GOALAREA[3] - GOALAREA[2]) / 2.0
#     # goal_center = [xGoal_center, yGoal_center, rrt_states[-1][2]]
#     # final_input = [0.0, 0.0]
#     # current_state = rrt_states[-1].reshape(3)
#     nmpc_states = np.array(nmpc_states).reshape(len(nmpc_states),num_states)
#     nmpc_ctrls = np.array(nmpc_ctrls).reshape(len(nmpc_ctrls), num_inputs)
#     distance_error = la.norm(final_state[0:2] - nmpc_states[-1][0:2])
#     print('Final error away from RRT* goal:', distance_error)
#
#     return nmpc_states, nmpc_ctrls

def disturbed_nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w):

    col_avoid = True

    # TODO: remove x_min, x_max, y_min, y_max from inputs
    _, env_edges = get_padded_edges()
    x_max = env_edges["right"]
    x_min = env_edges["left"]
    y_max = env_edges["top"]
    y_min = env_edges["bottom"]
    # Set up the Casadi solver
    if col_avoid:
        [solver, f, _, _, U, X, P, DELTA, STARTIDX] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)
    else:
        [solver, f, _, _, U, X, P, DELTA, STARTIDX] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)

    # # Fast solver without collision avoidance
    # [solverF, _, _, _, UF, XF, PF, _] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)
    # # Slower solver with collision avoidance
    # [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)

    final_input = [0.0, 0.0] # final input
    final_state = sim_state(T, rrt_states[-1].reshape(3), rrt_inputs[-1], f).full().reshape(3) # final state

    # pad rest of inputs and states with last state and last input for the rest of the horizon (N-1 times)
    rrt_inputs = rrt_inputs.tolist() # num_steps x num_controls (e.g. 200x2)
    rrt_states = rrt_states.tolist() # num_steps x num_states (e.g. 200x3)
    rrt_states.append(final_state) # first append the last state; now we have (num_steps+1) x num_states (e.g. 201x3)
    for _ in range(N-1):
        rrt_inputs.append(final_input)
        rrt_states.append(final_state)
    rrt_inputs = np.array(rrt_inputs)  # (num_steps+N-1) x num_controls (e.g. for N = 10: 209x2)
    rrt_states = np.array(rrt_states)  # (num_steps+1+N-1) x num_states (e.g. for N = 10: 210x3)

    ######################
    # Start NMPC Tracker #
    ######################
    #  Example: N = 2
    #                                                        repeat final input (= 0) N-1 times
    #                                                                              |
    #       (x0,u0)--->(x1,u1)--->(x2,u2)--->(x3,u3)--->(x4,u4)--->(x5,u5)--->(xT,u_f)--->(xT)
    #        |  |       |  |        |                                                       |
    #  current  |       |  |        |                                                repeat final states
    #  state    |_______|__|        |                                                    N-1 times
    #           |       |           |
    #      N horizon    |___________|
    #      next ref                 |
    #      inputs                   N horizon next ref states

    visited_states = []
    applied_controls = []
    current_state = rrt_states[0].reshape(num_states, 1)  # x0
    visited_states.append(current_state)  # mark current state as visited states

    # Note: num_steps = number of control steps available
    # The last control will take the system to the final state
    for itr in range(num_steps):
        current_state = visited_states[-1]  # Last visited state
        horizon_ref_states = rrt_states[itr+1:itr+N+1]  # next N rrt-planned states (N x num_states starting after current state)
        horizon_ref_inputs = rrt_inputs[itr:itr+N]  # next N rrt-planned inputs (N x num_controls starting at current state)
        current_goal_state = horizon_ref_states[-1].reshape(num_states, 1)  # end of current reference horizon states

        # index of node in horizon that with which collision avoidance should start
        # (use at least 1 to avoid crashes due to state realizations in collision zone)
        start_idx = max(1, N)

        # steer by solving NLP
        if col_avoid:
            x_casadi, u_casadi = nonlinsteerBigM(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
                                                 X, P, DELTA, STARTIDX, horizon_ref_states, horizon_ref_inputs, start_idx)
        else:
            x_casadi, u_casadi = nonlinsteerNoColAvoid(solver, current_state, current_goal_state, num_states, num_inputs, N,
                                                       T, U,
                                                       X, P, DELTA, STARTIDX, horizon_ref_states, horizon_ref_inputs, start_idx)
        if x_casadi == []:
            print("nmpc failed at itr: ", itr)
            break
            # return [], []

        # NLP succeeded and trajectory found
        nmpc_input = u_casadi[0]  # input to apply at current state
        nmpc_next_state = x_casadi[1]  # next state after nmpc_input is applied

        # realized next state with noise
        realized_next_state = nmpc_next_state.reshape(num_states, 1) + w[itr].reshape(num_states, 1)

        # update the visited states and applied controls
        visited_states.append(realized_next_state)
        applied_controls.append(nmpc_input.reshape(num_inputs, 1))

    last_rrt_state = copy.copy(nmpc_next_state)

    print('Done with nmpc')
    # # add a final while loop to keep robot at center of goal region
    # xGoal_center = GOALAREA[0] + (GOALAREA[1] - GOALAREA[0]) / 2.0
    # yGoal_center = GOALAREA[2] + (GOALAREA[3] - GOALAREA[2]) / 2.0
    # goal_center = [xGoal_center, yGoal_center, rrt_states[-1][2]]
    # final_input = [0.0, 0.0]
    # current_state = rrt_states[-1].reshape(3)
    visited_states = np.array(visited_states).reshape(len(visited_states),num_states)
    applied_controls = np.array(applied_controls).reshape(len(applied_controls), num_inputs)
    distance_error = la.norm(final_state[0:2] - visited_states[-1][0:2])
    print('Final error away from RRT* goal:', distance_error)

    return visited_states, applied_controls, last_rrt_state

def get_padded_edges():
    '''
    Finds the left, right, top, and bottom padded edges for the obstacles and the environment
    Outputs:
        obs_edges = edges of obstacles in the form of a list where each element is a dictionary with "top","bottom", "right", and "left"
        env_edges = edges of environment in the form of a dictionary with "top","bottom", "right", and "left"
    obs_edges should be used as (x < "left") or (x > "right") or (y < "bottom") or (y > "top")
    env_edges should be used as (x > "left") and (x < "right") and (y > "bottom") and (y < "top")
    '''
    randArea1 = copy.copy(RANDAREA)  # [xmin,xmax,ymin,ymax]
    obstacleList1 = copy.copy(OBSTACLELIST)  # [ox,oy,wd,ht]

    # environment bounds
    xmin = randArea1[0]
    xmax = randArea1[1]
    ymin = randArea1[2]
    ymax = randArea1[3]
    # thickness of env edges (doesn't matter much, anything > 0  works)
    thickness = 0.1
    # original environment area - width and height
    width = xmax - xmin
    height = ymax - ymin

    env_edges = {"left": xmin+ROBRAD, "right": xmax-ROBRAD, "bottom": ymin+ROBRAD, "top": ymax-ROBRAD}  # environment edges
    obs_edges = []

    # # top, bottom, right, and left rectangles for the env edges
    # env_bottom = [xmin - thickness, ymin - thickness, width + 2 * thickness, thickness]
    # env_top = [xmin - thickness, ymax, width + 2 * thickness, thickness]
    # env_right = [xmax, ymin - thickness, thickness, height + 2 * thickness]
    # env_left = [xmin - thickness, ymin - thickness, thickness, height + 2 * thickness]
    #
    # obstacleList1.append(env_bottom)
    # obstacleList1.append(env_top)
    # obstacleList1.append(env_right)
    # obstacleList1.append(env_left)

    # add enough padding for obstacles for robot radius
    for obs in obstacleList1:
        xmin = obs[0] - ROBRAD
        xmax = xmin + obs[2] + (2 * ROBRAD)
        ymin = obs[1] - ROBRAD
        ymax = ymin + obs[3] + (2 * ROBRAD)
        # edges = {"left": xmin+ROBRAD, "right": xmax-ROBRAD, "bottom": ymin+ROBRAD, "top": ymax-ROBRAD}
        edges = {"left": xmin, "right": xmax, "bottom": ymin, "top": ymax}
        obs_edges.append(edges)

    return obs_edges, env_edges

def get_state_bounds(obs_edges, env_edges, state):
    '''
    Finds the position bounds on a state given a set of obstacles (find maximum padding along each direction before
    colliding with an obstacle)
    '''

    eps = 0.00001 # arbitrarily small value

    # current state
    x = state[0]
    y = state[1]

    # environment bounds
    x_max_env = env_edges["right"]
    x_min_env = env_edges["left"]
    y_max_env = env_edges["top"]
    y_min_env = env_edges["bottom"]

    # lists to add upper and lower bounds for x and y
    # (min/max element selected from them later as the upper/lower bound)
    # Initialize them with environment bounds
    x_max_bounds = [x_max_env]
    x_min_bounds = [x_min_env]
    y_max_bounds = [y_max_env]
    y_min_bounds = [y_min_env]

    inside_obs_counter = 0 # check if state is inside multiple obstacles
    for obs_num, obs in enumerate(obs_edges):
        # obstacles
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        # TODO: This mess needs to be fixed. All obstacles need to be considered at once when the state could be inside
        #  an obstacle or when the state can be moved which might put it in an obstacle
        # if the state is inside the obstacle, we have to move the state outside (to the closest bound)
        if (left <= x <= right) and (bottom <= y <= top):
            inside_obs_counter += 1

            dr = abs(x - right) # x distance to right edge
            dl = abs(x - left) # x distance to left edge
            dt = abs(y - top) # y distance to top edge
            db = abs(y - bottom) # y distance to bottom edge

            d_list = [dr, dl, dt, db] # list of distances: right, left, top, bottom
            idx = d_list.index(min(d_list)) # index of closest distance: 0-->right, 1-->left, 2-->top, 3-->bottom

            if idx == 0:
                x_min_bounds.append(right) # right edge is closest --> make right edge a lower bound for x
                x = right + eps # move x to right edge
            elif idx == 1:
                x_max_bounds.append(left) # left edge is closest --> make left edge an upper bound for x
                x = left - eps # move x to left edge
            elif idx == 2:
                y_min_bounds.append(top) # top edge is closest --> make top edge a lower bound for y
                y = top + eps # move y to top edge
            elif idx == 3:
                y_max_bounds.append(bottom) # bottom edge is closest --> make bottom edge an upper bound for y
                y = bottom - eps  # move y to bottom edge
            else:
                print('ERROR: something is wrong')

            # if drl < dtb: # state closer to right or left edge
            #     if dr < dl: # if x is closer to the right edge, add right edge as x lower bound
            #         x_min_bounds.append(right)
            #     else:  # if x is closer to the left edge, add left edge as x upper bound
            #         x_max_bounds.append(left)
            # else: # state closer to top or bottom edge
            #     if dt < db:  # if y is closer to the top edge, add top edge as y lower bound
            #         y_min_bounds.append(top)
            #     else:  # if y is closer to the bottom edge, add bottom edge as y upper bound
            #         y_max_bounds.append(bottom)

        else: # state not inside an obstacle
            # add left edge of obstacle to x upper bounds if current state is to the left of the obstacle
            if (bottom <= y <= top) and (x <= left):
                x_max_bounds.append(left)

            # add right edge of obstacle to x lower bounds if current state is to the right of the obstacle
            if (bottom <= y <= top) and (x >= right):
                x_min_bounds.append(right)

            # add bottom edge of obstacle to y upper bounds if current state is to the bottom of the obstacle
            if (left <= x <= right) and (y <= bottom):
                y_max_bounds.append(bottom)

            # add top edge of obstacle to y lower bounds if current state is to the top of the obstacle
            if (left <= x <= right) and (y >= top):
                y_min_bounds.append(top)


    # find maximum lower bound and minimum upper bound
    xmax = min(x_max_bounds)
    xmin = max(x_min_bounds)
    ymax = min(y_max_bounds)
    ymin = max(y_min_bounds)

    for obs_num, obs in enumerate(obs_edges):
        # obstacles
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        # TODO: This mess needs to be fixed. All obstacles need to be considered at once when the state could be inside
        #  an obstacle or when the state can be moved which might put it in an obstacle
        # if the state is inside the obstacle, we have to move the state outside (to the closest bound)
        if (left <= x <= right) and (bottom <= y <= top):
            inside_obs_counter += 1

    if inside_obs_counter > 1:
        print('......................................................')
        print('ERROR: INSIDE MULTIPLE OBSTACLES. THIS IS NOT RESOLVED')
        print('******************************************************')
        return []


    return [xmin, xmax, ymin, ymax]

###############################################################################
####################### FUNCTION CALLED BY MAIN() #############################
###############################################################################

#TODO:change this to support different min and max values
def drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max, omega_max, x_max, y_max, theta_max, w=[],
                        save_plot = False,  save_file_name = ""):
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
    t_start = time.time()
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

    if w == []: # no disturbance
        # nmpc with no disturbance
        all_states_cl, all_inputs_cl, last_rrt_state = nmpc(nmpc_horizon, DT, v_max, -v_max, omega_max, -omega_max,
                                                            x_max, -x_max, y_max,
                                                            -y_max, theta_max, -theta_max,
                                                            rrt_states, rrt_inputs, GOALAREA, num_steps, n, m)
    else:
        # run nmpc with disturbance
        all_states_cl, all_inputs_cl, last_rrt_state = disturbed_nmpc(nmpc_horizon, DT, v_max, -v_max,
                                                                      omega_max, -omega_max,
                                                                      x_max, -x_max, y_max, -y_max, theta_max,
                                                                      -theta_max,
                                                                      rrt_states, rrt_inputs, GOALAREA,
                                                                      num_steps, n, m, w)
    # if it failed, stop
    if all_states_cl == []:
        print('Failed')
        return

    t_end = time.time()
    print('Total time: ', t_end - t_start)

    # extract the x and y states in the rrt plan
    x_orig = np.array(rrt_states).reshape(num_steps, n)[:, 0]
    x_orig = list(x_orig)
    x_orig.append(last_rrt_state[0])
    x_orig = np.array(x_orig)
    y_orig = np.array(rrt_states).reshape(num_steps, n)[:, 1]
    y_orig = list(y_orig)
    y_orig.append(last_rrt_state[1])
    y_orig = np.array(y_orig)

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

    return all_states_cl, all_inputs_cl, last_rrt_state


# TODO: main is no longer up to date MUST UPDATE
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
    all_states_cl, all_inputs_cl = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max, omega_max, x_max,
                                                     y_max, theta_max, w=[],
                                                     animate_results=True, save_plot=False, save_file_name=input_file,
                                                     ax_lim=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w,
                                                     wheel_h=wheel_h)
    animate(t_hist, all_states_cl, all_inputs_cl, x_ref_hist, u_ref_hist,
            title='NMPC, Closed-loop, reference',
            fig_offset=(1000, 400),
            axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)


    # simulate with disturbance
    all_states_cl_dist, all_inputs_cl_dist = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, v_max,
                                                               omega_max, x_max, y_max, theta_max, w=w_base_hist,
                                                               animate_results=True, save_plot=False,
                                                               save_file_name=input_file,
                                                               ax_lim=ax_lim, robot_w=robot_w, robot_h=robot_h,
                                                               wheel_w=wheel_w, wheel_h=wheel_h)

    animate(t_hist, all_states_cl_dist, all_inputs_cl_dist, x_ref_hist, u_ref_hist,
            title='NMPC, Closed-loop, referdisturbedence',
            fig_offset=(1000, 400),
            axis_limits=ax_lim, robot_w=robot_w, robot_h=robot_h, wheel_w=wheel_w, wheel_h=wheel_h)