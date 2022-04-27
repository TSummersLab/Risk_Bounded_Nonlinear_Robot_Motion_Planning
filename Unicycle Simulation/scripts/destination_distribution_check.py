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
import os
from plotting import animate
import copy

from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import EllipseCollection
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox

from collision_check import *

from drrrts_nmpc import SetUpSteeringLawParametersBigM
from drrrts_nmpc import nonlinsteerBigM, find_dr_padding, get_padded_edges
from drrrts_nmpc import disturbed_nmpc

import sys
sys.path.insert(0, '../unicycle')
sys.path.insert(0, '../rrtstar')
sys.path.insert(0, '../')
import UKF_Estimator as UKF_Estimator
from numpy import linalg as LA

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
SIGMAW = config.SIGMAW
SIGMAV = config.SIGMAV
CROSSCOR = config.CROSSCOR
ALFA = copy.copy(config.ALFA)
QLL = config.QLL
RLL = config.RLL
QTLL = config.QTLL
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse

# copy last alfa (env alfa) to beginning and remove the last 4 (env) alfas from the end

lastalfa = ALFA[-1]
obsalfa = ALFA[0:-4]
obsalfa.insert(0, lastalfa)
ALFA = obsalfa

def sim_state(T, x0, u, f):
    f_value = f(x0, u)
    st = x0+T * f_value.T
    return st

############################# NMPC FUNCTIONS ##################################
#
# def SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min):
#     """
#     Sets up an IPOPT NLP solver using Casadi Opti
#     Inputs:
#         N: horizon
#         T: time step (sec)
#         v_max, v_min: maximum and minimum linear velocities in m/s
#         omega_max, omega_min: maximum and minimum angular velocities in rad/s
#     Outputs:
#     solver, f, n_states, n_controls, U, X, P, delta
#         solver: Casadi NLP solver using bonmin
#         f: Casadi continuous time dynamics function
#         n_states, n_controls: number of states and controls
#         U, X: Casadi input and state variables (N x n_controls and (N+1)x n_states matrices)
#         P: Casadi desired state parameters ((N+1) x n_states matrix)
#         Delta: Casadi 0-1 variables for constraints (4*num_obs vector)
#     """
#
#     # Define state and input cost matrices
#     Q = QLL
#     R = RLL
#     QT = QTLL
#
#     opti = casadi.Opti()
#
#     # Define symbolic states using Casadi Opti
#     x = opti.variable()
#     y = opti.variable()
#     theta = opti.variable()
#     states = vertcat(x, y, theta)  # all three states
#     n_states = states.size()[0]  # number of symbolic states
#
#     # Define symbolic inputs using Cadadi SX
#     v = opti.variable()
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
#     # Casadi Opti trajectory variables/parameters for multiple shooting
#     U = opti.variable(N, n_controls)
#     X = opti.variable(N+1, n_states)
#     P = opti.parameter(N+1, n_states)
#     discrete = [False]*(N*n_controls + (N+1)*n_states) # specify U and X to be continuous variables
#
#     # Cost function
#     obj = 0  # objective/cost
#     opti.subject_to(X[0, :].T == P[0, :].T)
#     for i in range(N):
#         # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
#         obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
#         obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T]) # quadratic penalty on deviation from reference state
#
#         # compute the next state from the dynamics
#         x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]
#
#         # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
#         opti.subject_to(X[i + 1, :].T == x_next_.T)
#
#     # we might not be able to get back to the original target goal state
#     # alternatively, we have a large penalty of being away from it
#     obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T])
#
#     # minimize this objective
#     opti.minimize(obj)
#
#     # state environment constraints
#     _, env_edges = get_padded_edges()
#     x_max_env = env_edges["right"]
#     x_min_env = env_edges["left"]
#     y_max_env = env_edges["top"]
#     y_min_env = env_edges["bottom"]
#     opti.subject_to(opti.bounded(x_min_env, X[:, 0], x_max_env))
#     opti.subject_to(opti.bounded(y_min_env, X[:, 1], y_max_env))
#     opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
#     # input constraints
#     opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
#     opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))
#
#     # create a dict of the discrete flags
#     args = dict(discrete=discrete)
#     # specify the solver
#     # opti.solver("bonmin", args)
#     opti.solver("ipopt", args)
#
#     solver = opti # solver instance to return
#
#     DELTA = []
#     STARTIDX = []
#     OBSPAD, ENVPAD = [], []
#     return solver, f, n_states, n_controls, U, X, P, DELTA, STARTIDX, OBSPAD, ENVPAD
#
# def SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min):
#     """
#     Sets up a BONMIN MINLP solver using Casadi Opti
#     Collision avoidance is encoded with Big-M formulation
#
#     Inputs:
#         N: horizon
#         T: time step (sec)
#         v_max, v_min: maximum and minimum linear velocities in m/s
#         omega_max, omega_min: maximum and minimum angular velocities in rad/s
#     Outputs:
#     solver, f, n_states, n_controls, U, X, P, delta
#         solver: Casadi NLP solver using bonmin
#         f: Casadi continuous time dynamics function
#         n_states, n_controls: number of states and controls
#         U, X: Casadi input and state variables (N x n_controls and (N+1)x n_states matrices)
#         P: Casadi desired state parameters ((N+1) x n_states matrix)
#         Delta: Casadi 0-1 variables for constraints (4*num_obs vector)
#     """
#
#     # Define state and input cost matrices
#     Q = QLL
#     R = RLL
#     QT = QTLL
#
#
#     opti = casadi.Opti()
#
#     # Define symbolic states using Casadi Opti
#     x = opti.variable()
#     y = opti.variable()
#     theta = opti.variable()
#     states = vertcat(x, y, theta)  # all three states
#     n_states = states.size()[0]  # number of symbolic states
#
#     # Define symbolic inputs using Cadadi SX
#     v = opti.variable()
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
#     # Casadi Opti trajectory variables/parameters for multiple shooting
#     U = opti.variable(N, n_controls)
#     X = opti.variable(N+1, n_states)
#     P = opti.parameter(N+1, n_states)
#     discrete = [False]*(N*n_controls + (N+1)*n_states) # specify U and X to be continuous variables
#
#     # Cost function
#     obj = 0  # objective/cost
#     opti.subject_to(X[0, :].T == P[0, :].T)
#     for i in range(N):
#         # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
#         obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
#         obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T]) # quadratic penalty on deviation from reference state
#
#         # compute the next state from the dynamics
#         x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]
#
#         # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
#         opti.subject_to(X[i + 1, :].T == x_next_.T)
#
#     # we might not be able to get back to the original target goal state
#     # alternatively, we have a large penalty of being away from it
#     obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T])
#
#     # minimize this objective
#     opti.minimize(obj)
#
#     # state environment constraints
#     opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf)) # theta only now (x,y states added later)
#     # input constraints
#     opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
#     opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))
#
#
#     # obstacle constraints using Big-M formulation TODO: TRY THE CONVEX-HULL REFORMULATION https://optimization.mccormick.northwestern.edu/index.php/Disjunctive_inequalities (it might be faster)
#     obs_edges, env_edges = get_padded_edges()
#     x_max_env = env_edges["right"]
#     x_min_env = env_edges["left"]
#     y_max_env = env_edges["top"]
#     y_min_env = env_edges["bottom"]
#
#     num_obs = len(obs_edges)
#     DELTA = opti.variable(4 * num_obs) # 0-1 variables to indicate if an obstacle is hit
#     opti.subject_to(opti.bounded(0, DELTA, 1))
#     discrete += [True] * (4 * num_obs) # specify the delta variables to be discrete (with above bound --> 0-1 variables)
#     M = max(x_max_env-x_min_env, y_max_env-y_min_env) + 1 # 10 # a large upper bound on x and y
#     STARTIDX = opti.parameter(1)  # specify which points in the horizon should have collision avoidance enforced
#     # DR padding values
#     OBSPAD = opti.parameter(N+1, 4 * num_obs) # for each time step, each obstacle edge has its own dr padding (right, left, top, bottom)
#     ENVPAD = opti.parameter(N+1, 4) # for each time step, the four environment edges have their own dr padding (xmax, xmin, ymax, ymin) = (right, left, top, bottom)
#
#     opti.subject_to(opti.bounded(x_min_env + ENVPAD[:,1], X[:, 0], x_max_env - ENVPAD[:,0]))
#     opti.subject_to(opti.bounded(y_min_env + ENVPAD[:,3], X[:, 1], y_max_env - ENVPAD[:,2]))
#
#     for obs_num, obs in enumerate(obs_edges):
#         # for every obstacle
#         top = obs["top"]
#         bottom = obs["bottom"]
#         right = obs["right"]
#         left = obs["left"]
#
#         # add Big-M formulation disjunctive constraints
#         opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 0]) + right + OBSPAD[:, 0],
#                                      X[:, 0],
#                                      x_max_env - ENVPAD[:, 0] + M * (1 - DELTA[4 * obs_num + 0])))  # be to the right of the obstacle
#         opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 1]) + x_min_env + ENVPAD[:, 1],
#                                      X[:, 0],
#                                      left - OBSPAD[:, 1] + M * (1 - DELTA[4 * obs_num + 1])))  # be to the left of the obstacle
#         opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 2]) + top + OBSPAD[:, 2],
#                                      X[:, 1],
#                                      y_max_env - ENVPAD[:, 2] + M * (1 - DELTA[4 * obs_num + 2])))  # be to the top of the obstacle
#         opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 3]) + y_min_env + ENVPAD[:, 3],
#                                      X[:, 1],
#                                      bottom - OBSPAD[:, 3] + M * (1 - DELTA[4 * obs_num + 3])))  # be to the bottom of the obstacle
#
#         # require at least one of these constraints to be true
#         opti.subject_to(
#             1 <= DELTA[4 * obs_num + 0] + DELTA[4 * obs_num + 1] + DELTA[4 * obs_num + 2] + DELTA[4 * obs_num + 3])
#
#     # create a dict of the discrete flags
#     args = dict(discrete=discrete)
#     # specify the solver
#     opti.solver("bonmin", args)
#
#     solver = opti # solver instance to return
#
#     return solver, f, n_states, n_controls, U, X, P, DELTA, STARTIDX, OBSPAD, ENVPAD
#
# def nonlinsteerNoColAvoid(solver, x0, xT, n_states, n_controls, N, T, U, X, P, DELTA, STARTIDX, OBSPAD, ENVPAD, current_ref_traj, current_ref_inputs, start_idx, obs_pad, env_pad):
#     """
#     Solves the nonlinear steering problem using the solver from SetUpSteeringLawParametersBigM
#     Inputs:
#         solver: Casadi NLP solver from SetUpSteeringLawParameters
#         x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
#         n_states, n_controls: number of states and controls
#         N: horizon
#         T: time step
#         lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
#         current_ref_traj, current_ref_inputs: reference trajectory and reference inputs as Nx(n_states) ndarrays# TODO: add shapes
#     Outputs:
#         x_casadi, u_casadi: trajectory states and inputs returned by Casadi
#             if solution found:
#                 states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
#                 controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
#             else, [],[] returned
#     """
#
#     # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
#     init_states_param = np.linspace(0, 1, N + 1)
#     init_states = np.zeros([N + 1, n_states])
#     dx = xT - x0
#     for i in range(N + 1):
#         init_states[i] = (x0 + init_states_param[i] * dx).flatten()
#
#     # Create an initial input trajectory that roughly accomplishes the desired state transfer
#     # (using interpolated states to compute rough estimate of controls)
#     dist = la.norm(xT[0:2] - x0[0:2])
#     ang_dist = xT[2][0] - x0[2][0]
#     total_time = N * T
#     const_vel = dist / total_time
#     const_ang_vel = ang_dist / total_time
#     init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)
#
#     ## set parameter
#     constraint_states = []
#     constraint_states.append(x0.reshape(n_states))
#
#     for ref_state in current_ref_traj:
#         constraint_states.append(ref_state.reshape(n_states))
#
#     init_inputs = []
#     for ref_input in current_ref_inputs:
#         init_inputs.append(ref_input.reshape(n_controls))
#     init_inputs = np.array(init_inputs)
#
#     constraint_states = np.array(constraint_states)
#     solver.set_value(P, constraint_states)
#     solver.set_initial(X, constraint_states)
#     solver.set_initial(U, init_inputs)
#     # solver.set_initial(X, init_states)
#     # solver.set_initial(U, init_inputs)
#     try:
#         res = solver.solve()
#     except:
#         print('Steering NLP Failed')
#         return [], []
#
#     # Update the cost_total
#     # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
#     # Obtain the optimal control input sequence
#     u_casadi = res.value(U) # shape: (N, n_controls)
#     # Get the predicted state trajectory for N time steps ahead
#     x_casadi = res.value(X) # shape: # (N+1, n_states)
#
#     return x_casadi, u_casadi
#
# def nonlinsteerBigM(solver, x0, xT, n_states, n_controls, N, T, U, X, P, DELTA, STARTIDX, OBSPAD, ENVPAD, current_ref_traj, current_ref_inputs, start_idx, obs_pad, env_pad):
#     """
#     Solves the nonlinear steering problem using the solver from SetUpSteeringLawParametersBigM
#     Inputs:
#         solver: Casadi NLP solver from SetUpSteeringLawParameters
#         x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
#         n_states, n_controls: number of states and controls
#         N: horizon
#         T: time step
#         lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
#         current_ref_traj, current_ref_inputs: reference trajectory and reference inputs as Nx(n_states) ndarrays# TODO: add shapes
#     Outputs:
#         x_casadi, u_casadi: trajectory states and inputs returned by Casadi
#             if solution found:
#                 states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
#                 controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
#             else, [],[] returned
#     """
#
#     # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
#     init_states_param = np.linspace(0, 1, N + 1)
#     init_states = np.zeros([N + 1, n_states])
#     dx = xT - x0
#     for i in range(N + 1):
#         init_states[i] = (x0 + init_states_param[i] * dx).flatten()
#
#     # Create an initial input trajectory that roughly accomplishes the desired state transfer
#     # (using interpolated states to compute rough estimate of controls)
#     dist = la.norm(xT[0:2] - x0[0:2])
#     ang_dist = xT[2][0] - x0[2][0]
#     total_time = N * T
#     const_vel = dist / total_time
#     const_ang_vel = ang_dist / total_time
#     init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)
#
#     ## set parameter
#     constraint_states = []
#     constraint_states.append(x0.reshape(n_states))
#
#
#     for ref_state in current_ref_traj:
#         constraint_states.append(ref_state.reshape(n_states))
#     constraint_states = np.array(constraint_states)
#
#     init_inputs = []
#     for ref_input in current_ref_inputs:
#         init_inputs.append(ref_input.reshape(n_controls))
#     init_inputs = np.array(init_inputs)
#
#     solver.set_value(P, constraint_states)
#     solver.set_value(STARTIDX, start_idx)
#     solver.set_value(OBSPAD, obs_pad)
#     solver.set_value(ENVPAD, env_pad)
#     solver.set_initial(X, constraint_states)
#     solver.set_initial(U, init_inputs)
#     try:
#         res = solver.solve()
#     except:
#         print('Steering NLP Failed')
#         return [], []
#
#     # Update the cost_total
#     # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
#     # Obtain the optimal control input sequence
#     u_casadi = res.value(U)  # shape: (N, n_controls)
#     # Get the predicted state trajectory for N time steps ahead
#     x_casadi = res.value(X)  # shape: # (N+1, n_states)
#
#     print('delta', res.value(DELTA))
#
#     return x_casadi, u_casadi

def nmpc(N,T, rrt_states, rrt_inputs, num_steps, num_states, num_inputs,
         obstaclelist, envbounds, drnmpc):
    w = np.zeros([num_steps, num_states])
    return disturbed_nmpc(N, T, rrt_states, rrt_inputs, num_steps,
                          num_states, num_inputs, w, obstaclelist, envbounds, drnmpc)

# def disturbed_nmpc(N,T, rrt_states, rrt_inputs, num_steps, num_states, num_inputs, w, obstaclelist, envbounds, drnmpc):
#
#     # if drnmpc --> use col avoidance pipeline, if not drnmpc --> just do no col avoid
#
#     v_max = VELMAX
#     v_min = VELMIN
#     omega_max = ANGVELMAX
#     omega_min = ANGVELMIN
#
#     # TODO: remove x_min, x_max, y_min, y_max from inputs
#     obs_edges, _ = get_padded_edges()
#     # Set up the Casadi solver
#
#     if drnmpc:
#         [solver, f, _, _, U, X, P, DELTA, STARTIDX, OBSPAD, ENVPAD] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)
#         # else:
#         [solverN, _, _, _, UN, XN, PN, DELTAN, STARTIDXN, OBSPADN, ENVPADN] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)
#     else:
#         [solverN, f, _, _, UN, XN, PN, DELTAN, STARTIDXN, OBSPADN, ENVPADN] = SetUpSteeringLawParametersNoColAvoid(N, T,
#                                                                                                                    v_max,
#                                                                                                                    v_min,
#                                                                                                                    omega_max,
#                                                                                                                    omega_min)
#
#     final_input = [0.0, 0.0] # final input
#     final_state = sim_state(T, rrt_states[-1].reshape(3), rrt_inputs[-1], f).full().reshape(3) # final state
#
#     # pad rest of inputs and states with last state and last input for the rest of the horizon (N-1 times)
#     rrt_inputs = rrt_inputs.tolist() # num_steps x num_controls (e.g. 200x2)
#     rrt_states = rrt_states.tolist() # num_steps x num_states (e.g. 200x3)
#     rrt_states.append(final_state) # first append the last state; now we have (num_steps+1) x num_states (e.g. 201x3)
#     for _ in range(N-1):
#         rrt_inputs.append(final_input)
#         rrt_states.append(final_state)
#     rrt_inputs = np.array(rrt_inputs)  # (num_steps+N-1) x num_controls (e.g. for N = 10: 209x2)
#     rrt_states = np.array(rrt_states)  # (num_steps+1+N-1) x num_states (e.g. for N = 10: 210x3)
#
#     ######################
#     # Start NMPC Tracker #
#     ######################
#     #  Example: N = 2
#     #                                                        repeat final input (= 0) N-1 times
#     #                                                                              |
#     #       (x0,u0)--->(x1,u1)--->(x2,u2)--->(x3,u3)--->(x4,u4)--->(x5,u5)--->(xT,u_f)--->(xT)
#     #        |  |       |  |        |                                                       |
#     #  current  |       |  |        |                                                repeat final states
#     #  state    |_______|__|        |                                                    N-1 times
#     #           |       |           |
#     #      N horizon    |___________|
#     #      next ref                 |
#     #      inputs                   N horizon next ref states
#
#     # flags for function to terminate
#     pt_obs_collision_detected = False
#     line_obs_collision_detected = False
#     nlp_failed_flag = False
#
#     visited_states = []
#     applied_controls = []
#     current_state = rrt_states[0].reshape(num_states, 1)  # x0
#     # check if current state is safe
#     collision_detected = PtObsColFlag(current_state, obstaclelist, envbounds, ROBRAD)
#     if collision_detected:
#         pt_obs_collision_detected = True
#         return pt_obs_collision_detected, line_obs_collision_detected, nlp_failed_flag, [], [], []
#
#     visited_states.append(current_state)  # mark current state as visited states
#
#     # set the same threshold for the environment and the obstacles; e.g. alfa = [env_alfa, obs1_alfa, ..., obs5_alfa]
#     alfa = ALFA
#
#     SigmaW = SIGMAW
#     SigmaV = SIGMAV
#     CrossCor = CROSSCOR
#
#     # Note: num_steps = number of control steps available
#     # The last control will take the system to the final state
#     all_nmpc_planned_states = []
#     for itr in range(num_steps):
#         current_state = visited_states[-1]  # Last visited state
#         horizon_ref_states = rrt_states[itr+1:itr+N+1]  # next N rrt-planned states (N x num_states starting after current state)
#         horizon_ref_inputs = rrt_inputs[itr:itr+N]  # next N rrt-planned inputs (N x num_controls starting at current state)
#         current_goal_state = horizon_ref_states[-1].reshape(num_states, 1)  # end of current reference horizon states
#
#         # find covariance for all but the first state in the horizon
#         # first state/current state is deterministic
#         # covar of second state/next state is just SigmaW
#         # (X[t+1] = f(X[t], U[t]) + W[t]; f(.,.) is deterministic since X[t] is realized)
#         if drnmpc:
#             horizon_covars = ukfCovars(list(horizon_ref_states), list(horizon_ref_inputs[1:]), N-1, num_states, num_states, SigmaW, SigmaV, CrossCor, SigmaW)
#
#             env_pad, obs_pad = find_dr_padding(alfa, N, obs_edges, horizon_covars)
#
#         # index of node in horizon that with which collision avoidance should start
#         # (use at least 1 to avoid crashes due to state realizations in collision zone)
#         start_idx = max(1, N)
#         # obs_pad = 0*np.ones([N+1, 4 * num_obs])
#         # env_pad = np.zeros([N+1, 4])
#
#         # steer by solving NLP
#         if drnmpc:
#             x_casadi, u_casadi = nonlinsteerBigM(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
#                                                  X, P, DELTA, STARTIDX, OBSPAD, ENVPAD, horizon_ref_states,
#                                                  horizon_ref_inputs, start_idx, obs_pad, env_pad)
#         else:
#             x_casadi = [] # to active the next condition and just do the no col avoid steering
#
#         if x_casadi == []:
#             obs_pad = []
#             env_pad = []
#             x_casadi, u_casadi = nonlinsteerNoColAvoid(solverN, current_state, current_goal_state, num_states,
#                                                        num_inputs, N,
#                                                        T, UN,
#                                                        XN, PN, DELTAN, STARTIDXN, OBSPADN, ENVPADN, horizon_ref_states,
#                                                        horizon_ref_inputs, start_idx, obs_pad, env_pad)
#             # print("###################################################")
#             # print("###################################################")
#             # print("###################################################")
#             # print("###################################################")
#             # print("############### Switched solvers ##################")
#             # print("###################################################")
#             # print("###################################################")
#             # print("###################################################")
#             # print("###################################################")
#         if x_casadi == []:
#             nlp_failed_flag = True
#             print("nmpc failed at itr: ", itr)
#             break
#
#         all_nmpc_planned_states.append(x_casadi)
#
#         # NLP succeeded and trajectory found
#         nmpc_input = u_casadi[0]  # input to apply at current state
#         nmpc_next_state = x_casadi[1]  # next state after nmpc_input is applied
#
#         # realized next state with noise
#         realized_next_state = nmpc_next_state.reshape(num_states, 1) + w[itr].reshape(num_states, 1)
#         # check if realized state is safe
#         collision_detected = PtObsColFlag(realized_next_state, obstaclelist, envbounds, ROBRAD)
#         if collision_detected:
#             pt_obs_collision_detected = True
#             break
#
#         # check if line connecting previous state and realized state is safe
#         collision_detected = LineObsColFlag(current_state, realized_next_state, obstaclelist, ROBRAD)
#         if collision_detected:
#             line_obs_collision_detected = True
#             break
#
#         # update the visited states and applied controls
#         visited_states.append(realized_next_state)
#         applied_controls.append(nmpc_input.reshape(num_inputs, 1))
#
#     realized_states = visited_states
#
#     print('Done with nmpc')
#     visited_states = np.array(visited_states).reshape(len(visited_states), num_states)
#     applied_controls = np.array(applied_controls).reshape(len(applied_controls), num_inputs)
#     distance_error = la.norm(final_state[0:2] - visited_states[-1][0:2])
#     print('Final error away from RRT* goal:', distance_error)
#
#     result_data = {'pt_obs_collision_detected': pt_obs_collision_detected,
#                    'line_obs_collision_detected': line_obs_collision_detected,
#                    'nlp_failed_flag': nlp_failed_flag,
#                    'visited_states': visited_states,
#                    'applied_controls': applied_controls,
#                    'all_nmpc_planned_states': all_nmpc_planned_states}
#
#     return result_data
#
# def get_padded_edges():
#     '''
#     Finds the left, right, top, and bottom padded (by robot radius) edges for the obstacles and the environment
#     Outputs:
#         obs_edges = edges of obstacles in the form of a list where each element is a dictionary with "top","bottom", "right", and "left"
#         env_edges = edges of environment in the form of a dictionary with "top","bottom", "right", and "left"
#     obs_edges should be used as (x < "left") or (x > "right") or (y < "bottom") or (y > "top")
#     env_edges should be used as (x > "left") and (x < "right") and (y > "bottom") and (y < "top")
#     '''
#     randArea1 = copy.copy(RANDAREA)  # [xmin,xmax,ymin,ymax]
#     obstacleList1 = copy.copy(OBSTACLELIST)  # [ox,oy,wd,ht]
#
#     # environment bounds
#     xmin = randArea1[0]
#     xmax = randArea1[1]
#     ymin = randArea1[2]
#     ymax = randArea1[3]
#     # thickness of env edges (doesn't matter much, anything > 0  works)
#     thickness = 0.1
#     # original environment area - width and height
#     width = xmax - xmin
#     height = ymax - ymin
#
#     env_edges = {"left": xmin+ROBRAD, "right": xmax-ROBRAD, "bottom": ymin+ROBRAD, "top": ymax-ROBRAD}  # environment edges
#     obs_edges = []
#
#     # add enough padding for obstacles for robot radius
#     for obs in obstacleList1:
#         xmin = obs[0] - ROBRAD
#         xmax = xmin + obs[2] + (2 * ROBRAD)
#         ymin = obs[1] - ROBRAD
#         ymax = ymin + obs[3] + (2 * ROBRAD)
#         edges = {"left": xmin, "right": xmax, "bottom": ymin, "top": ymax}
#         obs_edges.append(edges)
#
#     return obs_edges, env_edges

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

################# UKF #######################
def ukfCovars(xHist, uHist, N, numStates, numOutputs, SigmaW, SigmaV, CrossCor, start_node_covar):
    '''
    compute covariances at each state
    xHist: list of states (list: N+1 elements each with num_states elements)
    uHist: list of control inputs (list: N elements each with num_controls elements)
    N: horizon length
    numStates, numOutputs: number of states and outputs
    SigmaW, SigmaV, CrossCor = process noise covariance, measurement noise covariance, and cross covariance between them
    start_node_covar: covariance at the initial node
    '''
    ukf_params = {}
    ukf_params["n_x"] = numStates
    ukf_params["n_o"] = numOutputs
    ukf_params["SigmaW"] = SigmaW
    ukf_params["SigmaV"] = SigmaV
    ukf_params["CrossCor"] = CrossCor
    ukf_params["dT"] = DT

    # Find covariances
    SigmaE = start_node_covar  # covariance at initial/from node
    covarHist = [SigmaE]
    for k in range(0, N):
        x_hat = xHist[k]
        u_k = uHist[k]
        y_k = xHist[k+1] # (we assume perfect full state feedback so y = x)

        ukf_params["x_hat"] = x_hat
        ukf_params["u_k"] = u_k
        ukf_params["SigmaE"] = SigmaE
        ukf_params["y_k"] = y_k

        ukf_estimator = UKF_Estimator.UKF()  # initialize the state estimator
        estimator_output = ukf_estimator.Estimate(ukf_params)  # get the estimates
        SigmaE = estimator_output["SigmaE"] # Unbox the covariance
        covarHist.append(SigmaE.reshape(numStates, numStates))

    return covarHist

################ DR Padding ##################
#
# def find_dr_padding(alfa, N, obs_edges, horizon_covars):
#     '''
#     Finds DR padding value for each environment and obstacle edge
#     '''
#     xDir = np.array([1, 0, 0])  # x direction
#     yDir = np.array([0, 1, 0])  # x direction
#     num_obs = len(obs_edges)
#
#     env_pad = np.zeros([N + 1, 4])  # for each time step, the four environment edges have their own dr padding (right, left, top, bottom)
#     obs_pad = np.zeros([N + 1, 4 * num_obs])  # for each time step, each obstacle edge has its own dr padding (right, left, top, bottom)
#
#     # find tightening value for all alfa values delta = sqrt((1-alfa)/alfa)
#     alpha = np.array(alfa, float)
#     delta = (1-alpha) / alpha
#     delta = delta**(0.5)
#     print("##############################")
#     print(delta)
#
#     for n in range(1,N+1):  # skip the first time step (no DR padding there - it is already realized)
#         sigma = horizon_covars[n-1]  # this step's covariance
#
#         # environment dr padding
#         rl_pad = delta[0] * math.sqrt(xDir.T @ sigma @ xDir)  # padding along right/left direction
#         tb_pad = delta[0] * math.sqrt(yDir.T @ sigma @ yDir)  # padding along top/bottom direction
#         env_pad[n, 0] = rl_pad  # right
#         env_pad[n, 1] = rl_pad  # left
#         env_pad[n, 2] = tb_pad  # top
#         env_pad[n, 3] = tb_pad  # bottom
#
#         # obstacle padding
#         for ob in range(num_obs):  # for every obstacle, do the above
#             rl_pad = delta[ob+1] * math.sqrt(xDir.T @ sigma @ xDir)  # padding along right/left direction
#             tb_pad = delta[ob+1] * math.sqrt(yDir.T @ sigma @ yDir)  # padding along top/bottom direction
#             obs_pad[n, 4 * ob + 0] = rl_pad  # right
#             obs_pad[n, 4 * ob + 1] = rl_pad  # left
#             obs_pad[n, 4 * ob + 2] = tb_pad  # top
#             obs_pad[n, 4 * ob + 3] = tb_pad  # bottom
#
#     return env_pad, obs_pad

###############################################################################
####################### FUNCTION CALLED BY MAIN() #############################
###############################################################################

#TODO:change this to support different min and max values
def drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, num_steps, w=[],
                        save_plot = False,  save_file_name = "", drnmpc = True):
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

    plotting_on = False
    obstaclelist = copy.copy(OBSTACLELIST)
    envbounds = copy.copy(RANDAREA)
    robrad = ROBRAD

    t_start = time.time()
    # load inputs and states
    rrt_states = x_ref_hist
    rrt_inputs = u_ref_hist

    x0 = [-2.,3.,0.]
    xT = [-1.0, 3.0, 0.0]
    num_steps = 1
    nmpc_horizon = 1
    u_guess = [0.1, 0.0]

    rrt_states = [np.array(x0)]
    rrt_states.append(np.array(xT))
    rrt_inputs = [np.array(u_guess)]
    results_data = disturbed_nmpc(nmpc_horizon, DT, rrt_states, rrt_inputs, num_steps, n, m, w, obstaclelist, envbounds, drnmpc)

    pt_obs_collision_detected = results_data["pt_obs_collision_detected"]
    line_obs_collision_detected = results_data["line_obs_collision_detected"]
    nlp_failed_flag = results_data["nlp_failed_flag"]
    all_states_cl = results_data["visited_states"]
    all_inputs_cl = results_data["applied_controls"]
    all_nmpc_planned_states = results_data["all_nmpc_planned_states"]

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if pt_obs_collision_detected:
        print('Collision between realized point and an obstacle/environment')
    if line_obs_collision_detected:
        print('Collision between line connecting realized point and previous point with an obstacle')
    if nlp_failed_flag:
        print('NLP failed for some reason')

    crash_idx = -1 # index when NMPC failed completely (-1 --> didn't fail)
    if pt_obs_collision_detected or line_obs_collision_detected or nlp_failed_flag:
        crash_idx = len(all_states_cl)

        # get last visited
        last_state = all_states_cl[-1]

        # pad states with last one and ctrl with nothing until num_steps
        zero_ctrl = all_inputs_cl[-1] * 0
        all_states_cl = list(all_states_cl)
        all_inputs_cl = list(all_inputs_cl)
        for padding_steps in range(crash_idx, num_steps+1):
            all_states_cl.append(last_state)
            all_inputs_cl.append(zero_ctrl)

        all_states_cl = np.array(all_states_cl).reshape(num_steps+1, n)
        all_inputs_cl = np.array(all_inputs_cl).reshape(num_steps, m)

    t_end = time.time()
    print('Total time: ', t_end - t_start)

    # compute final state of rrt plan # TODO: this is overkill, fix later
    opti = casadi.Opti()
    x,y,theta = opti.variable(), opti.variable(), opti.variable()
    states = vertcat(x, y, theta)  # all three states
    v, omega = opti.variable(), opti.variable()
    controls = vertcat(v, omega)  # both controls
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
    xtm1 = rrt_states[-1, :]
    xtm1 = xtm1.reshape(1,3)
    utm1 = rrt_inputs[-1, :]
    utm1 = utm1.reshape(1,2)
    last_rrt_state = f(xtm1, utm1) * DT + xtm1

    # RRT* x,y states
    # extract the x and y states in the rrt plan with computed last state appended
    x_orig = np.array(rrt_states).reshape(num_steps, n)[:, 0]
    x_orig = list(x_orig)
    x_orig.append(last_rrt_state[0])
    x_orig = np.array(x_orig)
    y_orig = np.array(rrt_states).reshape(num_steps, n)[:, 1]
    y_orig = list(y_orig)
    y_orig.append(last_rrt_state[1])
    y_orig = np.array(y_orig)

    # NMPC Realized x,y states
    # get the x,y states of nmpc
    x_cl = np.array(all_states_cl)[:, 0]
    y_cl = np.array(all_states_cl)[:, 1]

    ##################################################################
    # PLOTTING
    if plotting_on:
        # environment rectangle bottom left and top right corners
        xmin_randarea = RANDAREA[0]
        xmax_randarea = RANDAREA[1]
        ymin_randarea = RANDAREA[2]
        ymax_randarea = RANDAREA[3]
        # thickness of env edges (doesn't matter much, anything > 0  works)
        thickness = 0.1
        # original environment area - width and height
        width_randarea = xmax_randarea - xmin_randarea
        height_randarea = ymax_randarea - ymin_randarea

        # top, bottom, right, and left rectangles for the env edges
        env_bottom = [xmin_randarea - thickness, ymin_randarea - thickness, width_randarea + 2 * thickness, thickness]
        env_top = [xmin_randarea - thickness, ymax_randarea, width_randarea + 2 * thickness, thickness]
        env_right = [xmax_randarea, ymin_randarea - thickness, thickness, height_randarea + 2 * thickness]
        env_left = [xmin_randarea - thickness, ymin_randarea - thickness, thickness, height_randarea + 2 * thickness]

        # add env as obstacle
        OBSTACLELIST.append(env_bottom)
        OBSTACLELIST.append(env_top)
        OBSTACLELIST.append(env_right)
        OBSTACLELIST.append(env_left)

        # Create figure
        fig = plt.figure(figsize=[9, 9])
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        # ax.axis('equal')
        plt.axis([-5.2, 5.2, -5.3, 5.3])

        # Plot the environment boundary
        xy, w, h = (-5.0, -5.0), 10.0, 10.0
        r = Rectangle(xy, w, h, fc='none', ec='gold', lw=1)
        offsetbox = AuxTransformBox(ax.transData)
        offsetbox.add_artist(r)
        ab = AnnotationBbox(offsetbox, (xy[0] + w / 2., xy[1] + w / 2.),
                            boxcoords="data", pad=0.52, fontsize=20,
                            bboxprops=dict(facecolor="none", edgecolor='k', lw=20))
        ax.add_artist(ab)

        # Change ticklabel font size
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)

        # rough estimate of DR padding
        xDir = np.array([1, 0, 0])  # x direction
        yDir = np.array([1, 0, 0])  # y direction
        alpha = ALFA[0]
        delta = (1-alpha)/alpha
        delta = delta ** 0.5
        xdrpad = delta * math.sqrt(xDir.T @ SIGMAW @ xDir)
        ydrpad = delta * math.sqrt(yDir.T @ SIGMAW @ yDir)
        # Plot the rectangle obstacles with DR padding
        obstacles = [Rectangle(xy=[ox - ROBRAD - xdrpad, oy - ROBRAD - ydrpad],
                               width=wd + 2 * ROBRAD + 2*xdrpad,
                               height=ht + 2 * ROBRAD + 2*ydrpad,
                               angle=0,
                               color="palegoldenrod") for (ox, oy, wd, ht) in OBSTACLELIST]
        for obstacle in obstacles:
            ax.add_artist(obstacle)

        # Plot the rectangle obstacles with robot radius padding
        obstacles = [Rectangle(xy=[ox - ROBRAD, oy - ROBRAD],
                               width=wd +2 * ROBRAD,
                               height=ht +2 * ROBRAD,
                               angle=0,
                               color="mistyrose") for (ox, oy, wd, ht) in OBSTACLELIST]
        for obstacle in obstacles:
            ax.add_artist(obstacle)

        # Plot the true rectangle obstacles
        obstacles = [Rectangle(xy=[ox, oy],  # add radius padding
                               width=wd,  # add radius padding
                               height=ht,  # add radius padding
                               angle=0,
                               color="k") for (ox, oy, wd, ht) in OBSTACLELIST]
        for obstacle in obstacles:
            ax.add_artist(obstacle)

        # plot RRT* sampled points
        plt.plot(x_orig, y_orig, 'o', color='gray')
        # plot NMPC realized points
        plt.plot(x_cl, y_cl, 'x', color='red')

        colorlist = ["blue", "green", "orangered", "purple", "lime", "coral"]
        num_colors = len(colorlist)
        for idx, nmpc_plan in enumerate(all_nmpc_planned_states):
            nmpc_plan_x = nmpc_plan[:,0]
            nmpc_plan_y = nmpc_plan[:,1]
            plt.plot(nmpc_plan_x, nmpc_plan_y, color=colorlist[idx%num_colors])

        if save_plot:
            plot_name = save_file_name + '_plot_nmpc.png'
            plt.savefig(plot_name)
        plt.show()

    # result_data = {'all_states_cl': all_states_cl,
    #                'all_inputs_cl': all_inputs_cl,
    #                'pt_obs_collision_detected': pt_obs_collision_detected,
    #                'line_obs_collision_detected': line_obs_collision_detected,
    #                'nlp_failed_flag': nlp_failed_flag,
    #                'crash_idx': crash_idx,
    #                'last_rrt_state': last_rrt_state}

    collision_flag = pt_obs_collision_detected or line_obs_collision_detected or nlp_failed_flag
    result_data = {'x_hist': all_states_cl,
                   'u_hist': all_inputs_cl,
                   'collision_flag': collision_flag,
                   'collision_idx': crash_idx}

    return result_data


# TODO: main is no longer up to date MUST UPDATE
if __name__ == '__main__':

    dist = "nrm"
    show_hist = False
    SigmaW = SIGMAW
    sigma1 = SigmaW[0, 0]  # first entry in SigmaW
    alfa = ALFA

    plotting_on = False
    obstaclelist = copy.copy(OBSTACLELIST)
    envbounds = copy.copy(RANDAREA)
    robrad = ROBRAD

    t_start = time.time()
    # load inputs and states

    # x0 = [-2., 3., 0.]
    # x1 = [-1.9, 3, 0]
    # xT = [-1.8, 3.0, 0.0]

    x0 = [-2., 3., 0.]
    x1 = [-1.9, 3, 0]
    x2 = [-1.8, 3, 0]
    x3 = [-1.7, 3, 0]
    x4 = [-1.6, 3, 0]
    x5 = [-1.5, 3, 0]
    x6 = [-1.4, 3, 0]
    x7 = [-1.3, 3, 0]
    x8 = [-1.2, 3, 0]
    x9 = [-1.1, 3, 0]
    x10 = [-1.0, 3, 0]
    x11 = [-1.1, 3, 0]
    x12 = [-1.2, 3, 0]
    x13 = [-1.3, 3, 0]
    x14 = [-1.4, 3, 0]
    x15 = [-1.5, 3, 0]
    x16 = [-1.6, 3, 0]
    x17 = [-1.7, 3, 0]
    x18 = [-1.8, 3, 0]
    x19 = [-1.9, 3, 0]


    # x0 = [-2., 3., 0.]
    # x1 = [-1.95, 3, 0]
    # x2 = [-1.9, 3, 0]
    # x3 = [-1.85, 3, 0]
    # x4 = [-1.8, 3, 0]
    # x5 = [-1.75, 3, 0]
    # x6 = [-1.7, 3, 0]
    # x7 = [-1.65, 3, 0]
    # x8 = [-1.6, 3, 0]
    # x9 = [-1.55, 3, 0]
    # x10 = [-1.5, 3, 0]


    num_steps = 50
    T = num_steps
    nmpc_horizon = 5 #2
    u_guess = [2.5, 0.0]
    rrt_states = [np.array(x0)]
    rrt_states.append(np.array(x1))
    rrt_states.append(np.array(x2))
    rrt_states.append(np.array(x3))
    rrt_states.append(np.array(x4))
    rrt_states.append(np.array(x5))
    rrt_states.append(np.array(x6))
    rrt_states.append(np.array(x7))
    rrt_states.append(np.array(x8))
    rrt_states.append(np.array(x9))
    rrt_states.append(np.array(x10))
    rrt_states.append(np.array(x11))
    rrt_states.append(np.array(x12))
    rrt_states.append(np.array(x13))
    rrt_states.append(np.array(x14))
    rrt_states.append(np.array(x15))
    rrt_states.append(np.array(x16))
    rrt_states.append(np.array(x17))
    rrt_states.append(np.array(x18))
    rrt_states.append(np.array(x19))
    rrt_states.append(np.array(x0))
    rrt_states.append(np.array(x1))
    rrt_states.append(np.array(x2))
    rrt_states.append(np.array(x3))
    rrt_states.append(np.array(x4))
    rrt_states.append(np.array(x5))
    rrt_states.append(np.array(x6))
    rrt_states.append(np.array(x7))
    rrt_states.append(np.array(x8))
    rrt_states.append(np.array(x9))
    rrt_states.append(np.array(x10))
    rrt_states.append(np.array(x11))
    rrt_states.append(np.array(x12))
    rrt_states.append(np.array(x13))
    rrt_states.append(np.array(x14))
    rrt_states.append(np.array(x15))
    rrt_states.append(np.array(x16))
    rrt_states.append(np.array(x17))
    rrt_states.append(np.array(x18))
    rrt_states.append(np.array(x19))
    rrt_states.append(np.array(x0))
    rrt_states.append(np.array(x1))
    rrt_states.append(np.array(x2))
    rrt_states.append(np.array(x3))
    rrt_states.append(np.array(x4))
    rrt_states.append(np.array(x5))
    rrt_states.append(np.array(x6))
    rrt_states.append(np.array(x7))
    rrt_states.append(np.array(x8))
    rrt_states.append(np.array(x9))



    # rrt_states.append(np.array(xT))
    rrt_inputs = [np.array(u_guess)]*num_steps
    n, m = 3,2

    num_runs = 1000
    x_realized = []
    y_realized = []
    x2_realized = []
    y2_realized = []
    x3_realized = []
    y3_realized = []
    x4_realized = []
    y4_realized = []
    x5_realized = []
    y5_realized = []
    x6_realized = []
    y6_realized = []
    x7_realized = []
    y7_realized = []
    x8_realized = []
    y8_realized = []
    x9_realized = []
    y9_realized = []
    x10_realized = []
    y10_realized = []
    xT_realized = []
    yT_realized = []

    for i in range(num_runs):
        if dist == "nrm":
            w_base_hist = npr.multivariate_normal(mean=[0, 0, 0], cov=SigmaW, size=T)
            if show_hist:
                plt.hist(w_base_hist)
                plt.show()
        elif dist == "lap":
            l = 0
            b = (sigma1 / 2) ** 0.5
            w_base_hist = npr.laplace(loc=l, scale=b, size=[T, 3])  # mean = loc, var = 2*scale^2
            if show_hist:
                plt.hist(w_base_hist)
                plt.show()
        elif dist == "gum":
            b = (6 * sigma1) ** 0.5 / np.pi
            l = -0.57721 * b
            w_base_hist = npr.gumbel(loc=l, scale=b, size=[T, 3])  # mean = loc+0.57721*scale, var = pi^2/6 scale^2
            if show_hist:
                plt.hist(w_base_hist)
                plt.show()
        w = w_base_hist
        results_data = disturbed_nmpc(nmpc_horizon, DT, np.array(rrt_states), np.array(rrt_inputs), num_steps, n, m, w, obstaclelist, envbounds, drnmpc=True, hnmpc=True)
        # results_data = disturbed_nmpc(nmpc_horizon, DT, rrt_states, rrt_inputs, num_steps, n, m, w, obstaclelist, envbounds, drnmpc, hnmpc=hnmpc)

        pt_obs_collision_detected = results_data["pt_obs_collision_detected"]
        line_obs_collision_detected = results_data["line_obs_collision_detected"]
        nlp_failed_flag = results_data["nlp_failed_flag"]
        all_states_cl = results_data["visited_states"]
        all_inputs_cl = results_data["applied_controls"]
        all_nmpc_planned_states = results_data["all_nmpc_planned_states"]
        x_realized.append(all_states_cl[1][0])
        y_realized.append(all_states_cl[1][1])
        x2_realized.append(all_states_cl[2][0])
        y2_realized.append(all_states_cl[2][1])
        x3_realized.append(all_states_cl[3][0])
        y3_realized.append(all_states_cl[3][1])
        x4_realized.append(all_states_cl[4][0])
        y4_realized.append(all_states_cl[4][1])
        x5_realized.append(all_states_cl[5][0])
        y5_realized.append(all_states_cl[5][1])
        x6_realized.append(all_states_cl[6][0])
        y6_realized.append(all_states_cl[6][1])
        x7_realized.append(all_states_cl[7][0])
        y7_realized.append(all_states_cl[7][1])
        x8_realized.append(all_states_cl[8][0])
        y8_realized.append(all_states_cl[8][1])
        x9_realized.append(all_states_cl[9][0])
        y9_realized.append(all_states_cl[9][1])
        x10_realized.append(all_states_cl[10][0])
        y10_realized.append(all_states_cl[10][1])
        xT_realized.append(all_states_cl[-1][0])
        yT_realized.append(all_states_cl[-1][1])

    fig = plt.figure(figsize=[9, 9])
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # ax.axis('equal')
    plt.axis([-4, 0, -3, 4])

    # elE, elV = LA.eig(SigmaW[0:2, 0:2])
    # ellipse = Ellipse(xy=(x1[0], x1[1]), width=math.sqrt(elE[0])*2, height=math.sqrt(elE[1])*2,
    #                   edgecolor='b', fc='None', lw=2)
    # ax.add_patch(ellipse)
    # ellipse = Ellipse(xy=(x1[0], x1[1]), width=math.sqrt(elE[0])*6, height=math.sqrt(elE[1])*6,
    #                   edgecolor='b', fc='None', lw=2)
    # ax.add_patch(ellipse)
    # ellipse = Ellipse(xy=(xT[0], xT[1]), width=math.sqrt(elE[0]) * 6, height=math.sqrt(elE[1]) * 6,
    #                   edgecolor='b', fc='None', lw=2)
    # ax.add_patch(ellipse)

    xDir = np.array([1, 0, 0])  # x direction
    yDir = np.array([1, 0, 0])  # y direction
    # SIGMAW = np.diag([0.005, 0.005, 0.005])
    # alpha = 0.07
    alpha = alfa[0]
    delta = (1 - alpha) / alpha
    delta = delta ** 0.5
    xdrpad = delta * math.sqrt(xDir.T @ SIGMAW @ xDir) * 2
    ydrpad = delta * math.sqrt(yDir.T @ SIGMAW @ yDir) * 2


    # plot first step ellipse and its realizations
    ellipse = Ellipse(xy=(x1[0], x1[1]), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x_realized, y_realized, 'o', color='r', markersize=1)
    plt.plot(x0[0], x0[1], 'x', color='k')
    plt.plot(x1[0], x1[1], 'x', color='k')
    plt.plot(x2[0], x2[1], 'x', color='k')
    plt.plot(x3[0], x3[1], 'x', color='k')
    plt.plot(x4[0], x4[1], 'x', color='k')
    plt.plot(x5[0], x5[1], 'x', color='k')
    plt.plot(x6[0], x6[1], 'x', color='k')
    plt.plot(x7[0], x7[1], 'x', color='k')
    plt.plot(x8[0], x8[1], 'x', color='k')
    plt.plot(x9[0], x9[1], 'x', color='k')
    plt.plot(x10[0], x10[1], 'x', color='k')


    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -0.5
    ellipse = Ellipse(xy=(x2[0], x2[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x2_realized, list(np.array(y2_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -1
    ellipse = Ellipse(xy=(x3[0], x3[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x3_realized, list(np.array(y3_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -1.5
    ellipse = Ellipse(xy=(x4[0], x4[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x4_realized, list(np.array(y4_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -2
    ellipse = Ellipse(xy=(x5[0], x5[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x5_realized, list(np.array(y5_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -2.5
    ellipse = Ellipse(xy=(x6[0], x6[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x6_realized, list(np.array(y6_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -3
    ellipse = Ellipse(xy=(x7[0], x7[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x7_realized, list(np.array(y7_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -3.5
    ellipse = Ellipse(xy=(x8[0], x8[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x8_realized, list(np.array(y8_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -4
    ellipse = Ellipse(xy=(x9[0], x9[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x9_realized, list(np.array(y9_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -4.5
    ellipse = Ellipse(xy=(x10[0], x10[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(x10_realized, list(np.array(y10_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x0[0], x0[1] + shift_value, 'x', color='k')
    plt.plot(x1[0], x1[1] + shift_value, 'x', color='k')
    plt.plot(x2[0], x2[1] + shift_value, 'x', color='k')
    plt.plot(x3[0], x3[1] + shift_value, 'x', color='k')
    plt.plot(x4[0], x4[1] + shift_value, 'x', color='k')
    plt.plot(x5[0], x5[1] + shift_value, 'x', color='k')
    plt.plot(x6[0], x6[1] + shift_value, 'x', color='k')
    plt.plot(x7[0], x7[1] + shift_value, 'x', color='k')
    plt.plot(x8[0], x8[1] + shift_value, 'x', color='k')
    plt.plot(x9[0], x9[1] + shift_value, 'x', color='k')
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    # plot seconds step ellipse and its realizations (shift down by 1 for clarity)
    shift_value = -5
    ellipse = Ellipse(xy=(x10[0], x10[1] + shift_value), width=xdrpad, height=ydrpad,
                      edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.plot(xT_realized, list(np.array(yT_realized) + shift_value), 'o', color='g', markersize=1)
    plt.plot(x10[0], x10[1] + shift_value, 'x', color='k')

    plt.show()




