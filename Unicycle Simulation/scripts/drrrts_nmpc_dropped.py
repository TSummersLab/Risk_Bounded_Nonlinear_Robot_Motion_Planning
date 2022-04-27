#!/usr/bin/env python3
"""
Ideas I was trying but either didn't get working or stopped before completion
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

def SetUpSteeringLawParameters(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
    """
    Casadi SX + ipopt (no collision avoidance) -- original
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
    QT = 1000 * Q

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
    P = SX.sym('P', N + 1, n_states)  # all N+1 states as independent parameters

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
    obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T]) # TODO: Either keep this only without the equality constraint
    g.append(X[N, :].T - P[N, :].T)  # constraint on final state # TODO: or replace this by |X - P| < tolerance constraint

    # TODO: add obstacle constraints
    obs_edges, _ = get_padded_edges()

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


def SetUpSteeringLawParameters(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
    """
    casadi opti + bonmin + big-M obstacle avoidance

    Sets up an IPOPT NLP solver using Casadi Opti
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
    QT = 1000 * Q

    opti = casadi.Opti()

    # Define symbolic states using Casadi Opti
    # x = SX.sym('x') # x position
    x = opti.variable()
    # y = SX.sym('y') # y position
    y = opti.variable()
    # theta = SX.sym('theta') # heading angle
    theta = opti.variable()
    states = vertcat(x, y, theta)  # all three states
    n_states = states.size()[0]  # number of symbolic states

    # Define symbolic inputs using Cadadi SX
    # v = SX.sym('v') # linear velocity
    v = opti.variable()
    # omega = SX.sym('omega') # angular velocity
    omega = opti.variable()
    controls = vertcat(v, omega)  # both controls
    n_controls = controls.size()[0]  # number of symbolic inputs

    # RHS of nonlinear unicycle dynamics (continuous time model)
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)

    # Unicycle continuous time dynamics function
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # Casadi SX trajectory variables/parameters for multiple shooting
    # U = SX.sym('U', N, n_controls)  # N trajectory controls
    U = opti.variable(N, n_controls)
    # X = SX.sym('X', N + 1, n_states)  # N+1 trajectory states
    X = opti.variable(N+1, n_states)
    # P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters
    # P = SX.sym('P', N + 1, n_states)  # all N+1 states as independent parameters
    P = opti.parameter(N+1, n_states)

    discrete = [False]*(N*n_controls + (N+1)*n_states)

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    # g = []  # equality constraints
    # g.append(X[0, :].T - P[0, :].T)  # add constraint on initial state
    opti.subject_to(X[0, :].T == P[0, :].T)
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        # g.append(X[i + 1, :].T - x_next_.T)
        opti.subject_to(X[i + 1, :].T == x_next_.T)
    obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T]) # TODO: Either keep this only without the equality constraint
    opti.minimize(obj)
    # g.append(X[N, :].T - P[N, :].T)  # constraint on final state # TODO: or replace this by |X - P| < tolerance constraint
    # opti.subject_to(X[N, :].T == P[N, :].T) # TODO: uncomment ??? Maybe???

    # state environment constraints
    opti.subject_to(opti.bounded(x_min, X[:,0], x_max))
    opti.subject_to(opti.bounded(y_min, X[:,1], y_max))
    opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
    # input constraints
    opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
    opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))


    # # obstacle constraints TODO: add obstacle constraints
    # obs_edges, _ = get_padded_edges()
    # num_obs = len(obs_edges)
    # eps = 0.0001 # some small tolerance
    # delta = opti.variable(4*num_obs)
    # opti.subject_to(opti.bounded(0-eps, delta, 1+eps))
    # discrete += [True] * (4 * num_obs)
    # # for obs_num, obs in enumerate(obs_edges):
    # #     top = obs["top"]
    # #     bottom = obs["bottom"]
    # #     right = obs["right"]
    # #     left = obs["left"]
    # #     Ml = x_max - left + 1
    # #     ml = x_min - left - 1
    # #     Mr = x_max - right + 1
    # #     mr = x_min - right - 1
    # #     Mb = y_max - bottom - 1
    # #     mb = y_min - bottom - 1
    # #     Mt = y_max - top + 1
    # #     mt = y_min - top - 1
    # #
    # #     opti.subject_to(X[:, 0] + Ml * delta[4 * obs_num + 0] <= Ml + left)
    # #     opti.subject_to(X[:, 0] - (ml - eps) * delta[4 * obs_num + 0] >= left + eps)
    # #
    # #     opti.subject_to(X[:, 0] + mr * delta[4 * obs_num + 1] >= mr + right)
    # #     opti.subject_to(X[:, 0] - (Mr + eps) * delta[4 * obs_num + 1] <= right - eps)
    # #
    # #     opti.subject_to(X[:, 1] + Mb * delta[4 * obs_num + 2] <= Mb + bottom)
    # #     opti.subject_to(X[:, 1] - (mb - eps) * delta[4 * obs_num + 2] >= bottom + eps)
    # #
    # #     opti.subject_to(X[:, 1] + mt * delta[4 * obs_num + 3] >= mt + top)
    # #     opti.subject_to(X[:, 1] - (Mt + eps) * delta[4 * obs_num + 3] <= top - eps)
    # #
    # #     opti.subject_to(1-eps <= delta[4 * obs_num + 0] + delta[4 * obs_num + 1] + delta[4 * obs_num + 2] + delta[4 * obs_num + 3])

    # obstacle constraints using Big-M formulation TODO: TRY THE CONVEX-HULL REFORMULATION https://optimization.mccormick.northwestern.edu/index.php/Disjunctive_inequalities (it might be faster)
    # TODO: THIS BIG-M FORMULATION WORKS !!!!!!!!!!!!!!!!! YAY
    obs_edges, env_edges = get_padded_edges()
    x_max_env = env_edges["right"]
    x_min_env = env_edges["left"]
    y_max_env = env_edges["top"]
    y_min_env = env_edges["bottom"]

    num_obs = len(obs_edges)
    eps = 0.0001  # some small tolerance
    delta = opti.variable(4 * num_obs)
    opti.subject_to(opti.bounded(0 - eps, delta, 1 + eps))
    discrete += [True] * (4 * num_obs)
    M = 10 # a large upper bound on x and y
    for obs_num, obs in enumerate(obs_edges):
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 0]) + right, X[:, 0], x_max_env + M * (1 - delta[4 * obs_num + 0])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 1]) + x_min_env, X[:, 0], left + M * (1 - delta[4 * obs_num + 1])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 2]) + top, X[:, 1], y_max_env + M * (1 - delta[4 * obs_num + 2])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 3]) + y_min_env, X[:, 1], bottom + M * (1 - delta[4 * obs_num + 3])))

        opti.subject_to(
            1 <= delta[4 * obs_num + 0] + delta[4 * obs_num + 1] + delta[4 * obs_num + 2] + delta[4 * obs_num + 3])

    # Set the nlp problem settings
    opts_setting = {'ipopt.max_iter': 2000,
                    'ipopt.print_level': 4,  # 4
                    'print_time': 0,
                    'verbose': 1,  # 1
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6,
                    'error_on_fail': 1}
    args = dict(discrete=discrete) #dict(discrete=[False, False, True])

    # Set the nlp problem
    # nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    # opti.solver('ipopt', opts_setting)
    opti.solver("bonmin", args)
    solver = opti

    # # Create a solver that uses IPOPT with above solver settings
    # solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # # Define the bounds on states and controls
    # lbx = []
    # ubx = []
    # lbg = 0.0
    # ubg = 0.0
    # # Upper and lower bounds on controls
    # for _ in range(N):
    #     lbx.append(v_min)
    #     ubx.append(v_max)
    # for _ in range(N):
    #     lbx.append(omega_min)
    #     ubx.append(omega_max)
    # # Upper and lower bounds on states
    # for _ in range(N + 1):
    #     lbx.append(x_min)
    #     ubx.append(x_max)
    # for _ in range(N + 1):
    #     lbx.append(y_min)
    #     ubx.append(y_max)
    # for _ in range(N + 1):
    #     lbx.append(theta_min)
    #     ubx.append(theta_max)

    return solver, f, n_states, n_controls, U, X, P, delta


def SetUpSteeringLawParameters2(N, T, v_max, v_min, omega_max, omega_min, x_max=np.inf, x_min=-np.inf, y_max=np.inf, y_min=-np.inf, theta_max=np.inf, theta_min=-np.inf):
    """
    casadi opti + bonmin/ipopt + changing x,y bounds for collision avoidance

    Sets up an IPOPT NLP solver using Casadi Opti
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
    QT = 1000 * Q

    opti = casadi.Opti()

    # Define symbolic states using Casadi Opti
    # x = SX.sym('x') # x position
    x = opti.variable()
    # y = SX.sym('y') # y position
    y = opti.variable()
    # theta = SX.sym('theta') # heading angle
    theta = opti.variable()
    states = vertcat(x, y, theta)  # all three states
    n_states = states.size()[0]  # number of symbolic states

    # Define symbolic inputs using Cadadi SX
    # v = SX.sym('v') # linear velocity
    v = opti.variable()
    # omega = SX.sym('omega') # angular velocity
    omega = opti.variable()
    controls = vertcat(v, omega)  # both controls
    n_controls = controls.size()[0]  # number of symbolic inputs

    # RHS of nonlinear unicycle dynamics (continuous time model)
    rhs = horzcat(v * cos(theta), v * sin(theta), omega)

    # Unicycle continuous time dynamics function
    f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    # Casadi SX trajectory variables/parameters for multiple shooting
    # U = SX.sym('U', N, n_controls)  # N trajectory controls
    U = opti.variable(N, n_controls)
    # X = SX.sym('X', N + 1, n_states)  # N+1 trajectory states
    X = opti.variable(N+1, n_states)
    # P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters
    # P = SX.sym('P', N + 1, n_states)  # all N+1 states as independent parameters
    P = opti.parameter(N+1, n_states)

    # parameters for x,y state upper and lower bounds
    XMAX = opti.parameter(N + 1, 1)
    XMIN = opti.parameter(N + 1, 1)
    YMAX = opti.parameter(N + 1, 1)
    YMIN = opti.parameter(N + 1, 1)

    discrete = [False]*(N*n_controls + (N+1)*n_states)

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    # g = []  # equality constraints
    # g.append(X[0, :].T - P[0, :].T)  # add constraint on initial state
    opti.subject_to(X[0, :].T == P[0, :].T)
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        obj += mtimes([X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        # g.append(X[i + 1, :].T - x_next_.T)
        opti.subject_to(X[i + 1, :].T == x_next_.T)
    obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T]) # TODO: Either keep this only without the equality constraint
    opti.minimize(obj)
    # g.append(X[N, :].T - P[N, :].T)  # constraint on final state # TODO: or replace this by |X - P| < tolerance constraint
    # opti.subject_to(X[N, :].T == P[N, :].T) # TODO: uncomment ??? Maybe???

    # state environment constraints
    opti.subject_to(opti.bounded(XMIN, X[:,0], XMAX))
    opti.subject_to(opti.bounded(YMIN, X[:,1], YMAX))
    opti.subject_to(opti.bounded(-casadi.inf, X[:,2], casadi.inf))
    # input constraints
    opti.subject_to(opti.bounded(v_min, U[:,0], v_max))
    opti.subject_to(opti.bounded(omega_min, U[:,1], omega_max))


    # obstacle constraints TODO: add obstacle constraints
    obs_edges, _ = get_padded_edges()
    num_obs = len(obs_edges)
    eps = 0.0001 # some small tolerance
    delta = opti.variable(4*num_obs)
    opti.subject_to(opti.bounded(0-eps, delta, 1+eps))
    discrete += [True] * (4 * num_obs)

    # Set the nlp problem settings
    opts_setting = {'ipopt.max_iter': 2000,
                    'ipopt.print_level': 0,  # 4
                    'print_time': 0,
                    'verbose': 0,  # 1
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6,
                    'error_on_fail': 1}
    args = dict(discrete=discrete) #dict(discrete=[False, False, True])

    # Set the nlp problem
    # nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    # opti.solver('ipopt', opts_setting)
    opti.solver("bonmin", args)
    solver = opti

    return solver, f, n_states, n_controls, U, X, P, delta, XMIN, XMAX, YMIN, YMAX



def nonlinsteer3(solver, x0, xT, n_states, n_controls, N, T, U, X, P, delta, current_ref_traj, current_ref_inputs, obs_edges,
                                          env_edges, XMIN, XMAX, YMIN, YMAX, col_solver, XINIT, YINIT, DX, DY):
    """

    NLP solver using x,y bounds (V2)

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

    # Find upper and lower bounds for the state
    xmax = np.zeros([N + 1, 1])
    xmin = np.zeros([N + 1, 1])
    ymax = np.zeros([N + 1, 1])
    ymin = np.zeros([N + 1, 1])
    eps = 0.001
    for i in range(N + 1):
        state = init_states[i]
        xmin[i], xmax[i], ymin[i], ymax[i], xnew, ynew = get_state_bounds2(obs_edges, env_edges, state, col_solver, XINIT, YINIT, DX, DY)
        # print("xmin[i], xmax[i], ymin[i], ymax[i]", [xmin[i], xmax[i], ymin[i], ymax[i]])
        init_states[i, 0] = xnew
        init_states[i, 1] = ynew

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
    # init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    solver.set_value(P, constraint_states)
    solver.set_value(XMAX, xmax)
    solver.set_value(XMIN, xmin)
    solver.set_value(YMAX, ymax)
    solver.set_value(YMIN, ymin)
    solver.set_initial(X, init_states)
    solver.set_initial(U, init_inputs)

    try:
        res = solver.solve()
        # res = solver(x0=init_decision_vars, p=constraint_states, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    except:
        # raise Exception('NLP failed')
        print('NLP Failed')
        return [], []
    # res = solver.solve()

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    # casadi_result = res['x'].full()

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X)

    print('delta', res.value(delta))

    # Extract the control inputs and states
    # u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    # x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi


def nonlinsteer2(solver, x0, xT, n_states, n_controls, N, T, U, X, P, delta, current_ref_traj, current_ref_inputs, obs_edges,
                                          env_edges, XMIN, XMAX, YMIN, YMAX):
    """

    NLP solver using x,y bounds (V1)

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

    # Find upper and lower bounds for the state
    xmax = np.zeros([N + 1, 1])
    xmin = np.zeros([N + 1, 1])
    ymax = np.zeros([N + 1, 1])
    ymin = np.zeros([N + 1, 1])
    eps = 0.001
    for i in range(N + 1):
        state = init_states[i]
        xmin[i], xmax[i], ymin[i], ymax[i] = get_state_bounds(obs_edges, env_edges, state) # TODO: THIS WORKS BEST WITH SHORT HORIZON
        # print("xmin[i], xmax[i], ymin[i], ymax[i]", [xmin[i], xmax[i], ymin[i], ymax[i]])
        x = state[0]
        y = state[1]
        if i > 0:
            # if the initial state (from RRT plan) is not within permitted range, change it
            if not (xmin[i] <= x <= xmax[i]): # x not within limit
                d_to_min = abs(x - xmin[i])
                d_to_max = abs(x - xmax[i])
                if d_to_min < d_to_max: # set x to closer edge
                    init_states[i,0] = xmin[i] + eps
                else:
                    init_states[i,0] = xmax[i] - eps
            if not (ymin[i] <= y <= ymax[i]): # y not within limit
                d_to_min = abs(y - ymin[i])
                d_to_max = abs(y - ymax[i])
                if d_to_min < d_to_max: # set x to closer edge
                    init_states[i,1] = ymin[i] + eps
                else:
                    init_states[i,1] = ymax[i] - eps
            print("CHECK~~~~~~~")
            if not (xmin[i] <= init_states[i,0] <= xmax[i]) or not (ymin[i] <= init_states[i,1] <= ymax[i]):
                print('ERROR!!!!!!!!!!!!!!!!!!!!')

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
    # init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    solver.set_value(P, constraint_states)
    solver.set_value(XMAX, xmax)
    solver.set_value(XMIN, xmin)
    solver.set_value(YMAX, ymax)
    solver.set_value(YMIN, ymin)
    # solver.set_initial(X, init_decision_vars)
    solver.set_initial(X, init_states)
    solver.set_initial(U, init_inputs)
    try:
        res = solver.solve()
        # res = solver(x0=init_decision_vars, p=constraint_states, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    except:
        # raise Exception('NLP failed')
        print('NLP Failed')
        return [], []
    # res = solver.solve()

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    # casadi_result = res['x'].full()

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X)

    print('delta', res.value(delta))

    # Extract the control inputs and states
    # u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    # x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi


def nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, U, X, P, delta, current_ref_traj, current_ref_inputs):
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

    # Find upper and lower bounds for the state
    for i in range(N + 1):
        state = init_states[i]
        obs_edges, env_edges = get_padded_edges() # TODO: move this outside later
        xmin, xmax, ymin, ymax = get_state_bounds(obs_edges, env_edges, state)

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
    # init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    solver.set_value(P, constraint_states)
    # solver.set_initial(X, init_decision_vars)
    solver.set_initial(X, init_states)
    solver.set_initial(U, init_inputs)
    try:
        res = solver.solve()
        # res = solver(x0=init_decision_vars, p=constraint_states, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    except:
        # raise Exception('NLP failed')
        print('NLP Failed')
        return [], []
    # res = solver.solve()

    # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
    # casadi_result = res['x'].full()

    # Update the cost_total
    # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
    # Obtain the optimal control input sequence
    u_casadi = res.value(U)
    # Get the predicted state trajectory for N time steps ahead
    x_casadi = res.value(X)

    print('delta', res.value(delta))

    # Extract the control inputs and states
    # u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
    # x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

    return x_casadi, u_casadi

def nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx, current_ref_traj, current_ref_inputs):
    """

    original NLP solver (for casadi SX)

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


def disturbed_nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w):

    # TODO: remove x_min, x_max, y_min, y_max from inputs
    _, env_edges = get_padded_edges()
    x_max = env_edges["right"]
    x_min = env_edges["left"]
    y_max = env_edges["top"]
    y_min = env_edges["bottom"]
    # Set up the Casadi solver
    # [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)

    [solver, f, _, _, U, X, P, delta] = SetUpSteeringLawParameters(N, T, v_max, v_min,
                                                                       omega_max, omega_min,
                                                                       x_max, x_min,
                                                                       y_max, y_min,
                                                                       theta_max, theta_min)

    # [solver, f, _, _, U, X, P, delta, XMIN, XMAX, YMIN, YMAX] = SetUpSteeringLawParameters2(N, T, v_max, v_min,
    #                                                                                         omega_max, omega_min,
    #                                                                                         x_max, x_min,
    #                                                                                         y_max, y_min,
    #                                                                                         theta_max, theta_min)
    # col_solver, XINIT, YINIT, DX, DY = setup_moving_state()

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

        # x_casadi, u_casadi = nonlinsteerBigM(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
        #                                      X, P, DELTA, current_ref_traj, current_ref_inputs)

        x_casadi, u_casadi = nonlinsteer(solver, current_state, current_goal_state, num_states, num_inputs,
                                         N, T, U, X, P, delta, current_ref_traj, current_ref_inputs)

        # obs_edges, env_edges = get_padded_edges()
        # x_casadi, u_casadi = nonlinsteer2(solver, current_state, current_goal_state, num_states, num_inputs,
        #                                   N, T, U, X, P, delta, current_ref_traj, current_ref_inputs, obs_edges,
        #                                   env_edges, XMIN, XMAX, YMIN, YMAX)
        #
        # x_casadi, u_casadi = nonlinsteer3(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U, X,
        #                                   P, delta, current_ref_traj,
        #                                   current_ref_inputs, obs_edges,
        #                                   env_edges, XMIN, XMAX, YMIN, YMAX, col_solver, XINIT, YINIT, DX, DY)

        if x_casadi == []:
            print("nmpc failed at itr: ", itr)
            break
            # return [], []

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

# these might not be complete

def setup_moving_state():
    '''
    setup optimization problem to move state outside of obstacle set
    '''

    opti = casadi.Opti()
    DX = opti.variable() # distance to move along x direction
    DY = opti.variable() # distance to move along y direction
    discrete = [False]*2
    XINIT = opti.parameter() # original x position (to be initialized)
    YINIT = opti.parameter() # original y position (to be initialized)

    # opti.minimize(fabs(DX) + fabs(DY)) # minimize distance to be moved
    opti.minimize(DX**2 + DY**2)  # minimize distance to be moved

    # get obstacles and environment
    obs_edges, env_edges = get_padded_edges()
    x_max_env = env_edges["right"]
    x_min_env = env_edges["left"]
    y_max_env = env_edges["top"]
    y_min_env = env_edges["bottom"]

    # # constraints on new state (x+dx, y+dy) being in environment bounds
    # opti.subject_to(opti.bounded(x_min_env, x + dx, x_max_env))
    # opti.subject_to(opti.bounded(y_min_env, y + dx, y_max_env))
    # TODO: the above two constraints are already incorportated below

    # constraints on new state (x+dx, y+dy) being outside the obstacles and being in environment bounds
    num_obs = len(obs_edges)
    eps = 0.0001  # some small tolerance
    delta = opti.variable(4 * num_obs)
    opti.subject_to(opti.bounded(0 - eps, delta, 1 + eps))
    discrete += [True] * (4 * num_obs)
    M = 10 # max(abs(x_max_env - x_min_env), abs(y_max_env - y_min_env)) + 2  # a large upper bound on x and y
    for obs_num, obs in enumerate(obs_edges):
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 0]) + right, XINIT + DX,
                                     x_max_env + M * (1 - delta[4 * obs_num + 0])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 1]) + x_min_env, XINIT + DX,
                                     left + M * (1 - delta[4 * obs_num + 1])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 2]) + top, YINIT + DY,
                                     y_max_env + M * (1 - delta[4 * obs_num + 2])))
        opti.subject_to(opti.bounded(-M * (1 - delta[4 * obs_num + 3]) + y_min_env, YINIT + DY,
                                     bottom + M * (1 - delta[4 * obs_num + 3])))
        opti.subject_to(
            1 <= delta[4 * obs_num + 0] + delta[4 * obs_num + 1] + delta[4 * obs_num + 2] + delta[4 * obs_num + 3])

    args = dict(discrete=discrete)
    opti.solver("bonmin", args)
    col_solver = opti

    return col_solver, XINIT, YINIT, DX, DY

def move_state_outside_obstacles(col_solver, XINIT, YINIT, xinit, yinit, DX, DY):
    '''
    moves the state outside of the obstacles while keeping it in environmnet
    '''

    eps = 0.00001

    # initialize the original state
    col_solver.set_value(XINIT, xinit)
    col_solver.set_value(YINIT, yinit)

    # # TODO: maybe set initial value for dx, dy?
    col_solver.set_initial(DX, xinit - -4)
    col_solver.set_initial(DY, yinit - -4)

    # solve
    # try:
    print('Solving second NLP')
    res = col_solver.solve()
    # except:
    #     print('MINLP 2 Failed')
    #     return [], []

    dx_found = res.value(DX)
    dy_found = res.value(DY)

    if dx_found > 0:
        epsx = eps
    else:
        epsx = -eps

    if dy_found > 0:
        epsy = eps
    else:
        epsy = -eps

    return xinit + dx_found + epsx, yinit + dy_found + epsy

def get_state_bounds2(obs_edges, env_edges, state, col_solver, XINIT, YINIT, DX, DY):
    '''
    Move state and get state bounds
    '''
    # original state
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

    # check if state is in an obstacle
    in_obstacle = False  # check if state is inside an obstacle
    for obs_num, obs in enumerate(obs_edges):
        # obstacles
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        # if the state is inside the obstacle, we have to move the state outside (to the closest bound)
        if (left <= x <= right) and (bottom <= y <= top):
            in_obstacle = True
            break

    if in_obstacle: # if state in obstacle, move it out
        xnew, ynew = move_state_outside_obstacles(col_solver, XINIT, YINIT, x, y, DX, DY)
    else: # if state is not in any obstacle, continue
        xnew = x
        ynew = y

    # find bounds on x and y
    for obs_num, obs in enumerate(obs_edges):
        # obstacles
        top = obs["top"]
        bottom = obs["bottom"]
        right = obs["right"]
        left = obs["left"]

        # add left edge of obstacle to x upper bounds if current state is to the left of the obstacle
        if (bottom <= ynew <= top) and (xnew <= left):
            x_max_bounds.append(left)

        # add right edge of obstacle to x lower bounds if current state is to the right of the obstacle
        if (bottom <= ynew <= top) and (xnew >= right):
            x_min_bounds.append(right)

        # add bottom edge of obstacle to y upper bounds if current state is to the bottom of the obstacle
        if (left <= xnew <= right) and (ynew <= bottom):
            y_max_bounds.append(bottom)

        # add top edge of obstacle to y lower bounds if current state is to the top of the obstacle
        if (left <= xnew <= right) and (ynew >= top):
            y_min_bounds.append(top)

        # find maximum lower bound and minimum upper bound
    xmax = min(x_max_bounds)
    xmin = max(x_min_bounds)
    ymax = min(y_max_bounds)
    ymin = max(y_min_bounds)

    return [xmin, xmax, ymin, ymax, xnew, ynew]


class MPCTracker():
    """
    class for MPC low-level tracking
    """

    def __init__(self, N, T, v_min, v_max, omega_min, omega_max, code):
        """
        Constructor Function
        List of trackers:
        code(string): Solver/ Casadi Framework/ Collision avoidance
        IPOPT: 0         SX: 0          no collision avoidance: 0
        BONMIN: 1        Opti: 1        big-M formulation: 1

        "000": IPOPT/ Casadi SX/ no collision avoidance
        "110": BONMIN/ Casadi Opti/ no collision avoidance
        "111": BONMIN/ Casadi Opti/ Big-M formulation collision avoidance

        """
        self.N = N  # Tracker Horizon
        self.T = T  # Time step (sec)
        self.v_min = v_min  # minimum linear velocity
        self.v_max = v_max  # maximum linear velocity
        self.omega_min = omega_min  # minimum angular velocity
        self.omega_max = omega_max  # maximum angular velocity
        self.code = code  # Tracker type
        self.solver_type, self.ca_framework, self.col_avoid = self.ParseType()  # Parses the code input

        # Matrices for quadratic cost
        self.Q = 1000 * np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.1]])
        self.R = 1 * np.array([[1.0, 0.0],
                          [0.0, 1.0]])
        self.QT = 1000 * self.Q
        self.solver, self.f, self.n_states, self.n_controls, self.U, self.X, self.P, self.DELTA = self.SetupSteeringLawParams()

    def ParseType(self):
        solver_code = self.code[2]
        casadi_framework_code = self.code[1]
        collision_avoidance_code = self.code[0]
        # solver type
        if solver_code == '0':
            solver_type = "ipopt"
        elif solver_code == '1':
            solver_type = "bonmin"
        else:
            print('Solver not supported')
            solver_type = ''

        # Casadi Framework type
        if casadi_framework_code == '0':
            print('Not supported yet')
            ca_framework = "sx"
        elif casadi_framework_code == '1':
            ca_framework = "opti"
        else:
            print('Casadi Framework not supported')
            ca_framework = ''

        # Collision Avoidance type
        if collision_avoidance_code == '0':
            col_avoid = "none"
        elif collision_avoidance_code == '1':
            col_avoid = "bigM"
        else:
            print('Collision Avoidance not supported')
            col_avoid = ''

        return solver_type, ca_framework, col_avoid

    def SetupSteeringLawParams(self):

        # Cost matrix
        Q, R, QT = self.Q, self.R, self.QT
        # horizon, timestep, and control input bounds
        N, T, v_max, v_min, omega_max, omega_min = self.N, self.T, self.v_max, self.v_min, self.omega_max, self.omega_min
        # solver type, Casadi framework, and collision avoidance type
        solver_type, ca_framework, col_avoid = self.solver_type, self.ca_framework, self.col_avoid

        if ca_framework == "opti":
            opti = casadi.Opti()

            # Define symbolic states using Casadi Opti
            x = opti.variable()
            y = opti.variable()
            theta = opti.variable()

            # Define symbolic inputs using Cadadi Opti
            v = opti.variable()
            omega = opti.variable()
        else:
            print('Casadi Framework Error: Specified Framework NOT Supported')
            return

        # Define symbolic states using Casadi Opti
        states = vertcat(x, y, theta)  # all three states
        n_states = states.size()[0]  # number of symbolic states

        # Define symbolic inputs using Cadadi Opti
        controls = vertcat(v, omega)  # both controls
        n_controls = controls.size()[0]  # number of symbolic inputs

        # RHS of nonlinear unicycle dynamics (continuous time model)
        rhs = horzcat(v * cos(theta), v * sin(theta), omega)

        # Unicycle continuous time dynamics function
        f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

        if ca_framework == "opti":
            # Casadi Opti trajectory variables/parameters for multiple shooting
            U = opti.variable(N, n_controls)
            X = opti.variable(N + 1, n_states)
            P = opti.parameter(N + 1, n_states)
        else:
            print('Casadi Framework Error: NOT Supported yet')
            return

        if solver_type == "bonmin":
            # specify U and X to be continuous variables
            discrete = [False] * (N * n_controls + (N + 1) * n_states)
        else:
            print('Solver Type Error: NOT Supported yet')
            return


        # Cost function
        obj = 0  # objective/cost
        if ca_framework == "opti":
            opti.subject_to(X[0, :].T == P[0, :].T)
        else:
            print('Casadi Framework Error: NOT Supported yet')
            return
        # Add cost function terms
        for i in range(N):
            # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
            obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
            obj += mtimes(
                [X[i, :] - P[i, :], Q, X[i, :].T - P[i, :].T])  # quadratic penalty on deviation from reference state

            # compute the next state from the dynamics
            x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

            # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting) (satisfy dynamics)
            if ca_framework == "opti":
                opti.subject_to(X[i + 1, :].T == x_next_.T)
            else:
                print('Casadi Framework Error: NOT Supported yet')
                return

        # we might not be able to get back to the original target goal state
        # alternatively, we have a large penalty of being away from it
        obj += mtimes([X[N, :] - P[N, :], QT, X[N, :].T - P[N, :].T])

        # minimize this objective
        if ca_framework == "opti":
            opti.minimize(obj)
        else:
            print('Casadi Framework Error: NOT Supported yet')
            return

        # state environment constraints
        obs_edges, env_edges = get_padded_edges()
        x_max_env = env_edges["right"]
        x_min_env = env_edges["left"]
        y_max_env = env_edges["top"]
        y_min_env = env_edges["bottom"]
        if ca_framework == "opti":
            opti.subject_to(opti.bounded(x_min_env, X[:, 0], x_max_env))
            opti.subject_to(opti.bounded(y_min_env, X[:, 1], y_max_env))
            opti.subject_to(opti.bounded(-casadi.inf, X[:, 2], casadi.inf))
            # input constraints
            opti.subject_to(opti.bounded(v_min, U[:, 0], v_max))
            opti.subject_to(opti.bounded(omega_min, U[:, 1], omega_max))
        else:
            print('Casadi Framework Error: NOT Supported yet')
            return

        if col_avoid == "none":
            DELTA = []
        elif col_avoid == "bigM" and ca_framework == "opti":
            num_obs = len(obs_edges)
            DELTA = opti.variable(4 * num_obs)  # 0-1 variables to indicate if an obstacle is hit
            opti.subject_to(opti.bounded(0, DELTA, 1))
            discrete += [True] * (
                        4 * num_obs)  # specify the delta variables to be discrete (with above bound --> 0-1 variables)
            M = max(x_max_env - x_min_env, y_max_env - y_min_env) + 1  # 10 # a large upper bound on x and y
            for obs_num, obs in enumerate(obs_edges):
                # for every obstacle
                top = obs["top"]
                bottom = obs["bottom"]
                right = obs["right"]
                left = obs["left"]

                # add Big-M formulation disjunctive constraints
                opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 0]) + right, X[:, 0], x_max_env + M * (
                            1 - DELTA[4 * obs_num + 0])))  # be to the right of the obstacle
                opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 1]) + x_min_env, X[:, 0],
                                             left + M * (1 - DELTA[4 * obs_num + 1])))  # be to the left of the obstacle
                opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 2]) + top, X[:, 1], y_max_env + M * (
                            1 - DELTA[4 * obs_num + 2])))  # be to the top of the obstacle
                opti.subject_to(opti.bounded(-M * (1 - DELTA[4 * obs_num + 3]) + y_min_env, X[:, 1], bottom + M * (
                            1 - DELTA[4 * obs_num + 3])))  # be to the bottom of the obstacle

                # require at least one of these constraints to be true
                opti.subject_to(
                    1 <= DELTA[4 * obs_num + 0] + DELTA[4 * obs_num + 1] + DELTA[4 * obs_num + 2] + DELTA[
                        4 * obs_num + 3])
        else:
            print('Col Avoid not supported')
            return

        if solver_type == "bonmin":
            # create a dict of the discrete flags
            args = dict(discrete=discrete)
        else:
            print('Solver Type Error: NOT Supported yet')
            return

        if solver_type == "bonmin" and ca_framework == "opti":
            # specify the solver
            opti.solver("bonmin", args)
            solver = opti  # solver instance to return
        else:
            print('Solver/Framework Error: Not Supported combo')

        return solver, f, n_states, n_controls, U, X, P, DELTA

    def NonLinSteer(self, x0, xT, current_ref_traj, current_ref_inputs):
        # load params
        solver = self.solver
        n_states, n_controls = self.n_states, self.n_controls
        N, T, U, X, P, DELTA = self.N, self.T, self.U, self.X, self.P, self.DELTA

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

        if self.ca_framework == "opti":
            solver.set_value(P, constraint_states)
            solver.set_initial(X, init_states)
            solver.set_initial(U, init_inputs)
            try:
                res = solver.solve()
            except:
                print('Steering NLP Failed')
                return [], []

            # Update the cost_total
            # cost_total = res.value(self.obj)  # self.opti.debug.value(self.obj)
            # Obtain the optimal control input sequence
            u_casadi = res.value(U)  # shape: (N, n_controls)
            # Get the predicted state trajectory for N time steps ahead
            x_casadi = res.value(X)  # shape: # (N+1, n_states)

            if self.col_avoid == "bigM":
                print('delta', res.value(DELTA))
            else:
                print('Col Avoid Error: Not Supported Yet')

        else:
            print('Solver Type Error: Not Supported Yet')

        return x_casadi, u_casadi

def disturbed_nmpc(N,T,v_max, v_min, omega_max, omega_min, x_max, x_min, y_max, y_min, theta_max, theta_min,
         rrt_states, rrt_inputs, goal_area, num_steps, num_states, num_inputs, w):

    col_avoid = False

    # TODO: remove x_min, x_max, y_min, y_max from inputs
    _, env_edges = get_padded_edges()
    x_max = env_edges["right"]
    x_min = env_edges["left"]
    y_max = env_edges["top"]
    y_min = env_edges["bottom"]
    # Set up the Casadi solver
    if col_avoid:
        [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersBigM(N, T, v_max, v_min, omega_max, omega_min)
    else:
        [solver, f, _, _, U, X, P, DELTA] = SetUpSteeringLawParametersNoColAvoid(N, T, v_max, v_min, omega_max, omega_min)

    # if col_avoid:
    #     tracker = MPCTracker(N, T, v_min, v_max, omega_min, omega_max, code = "110")
    # else:
    #     tracker = MPCTracker(N, T, v_min, v_max, omega_min, omega_max, code = "111")
    # f = tracker.f

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

        if col_avoid:
            x_casadi, u_casadi = nonlinsteerBigM(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
                                                 X, P, DELTA, current_ref_traj, current_ref_inputs)
        else:
            x_casadi, u_casadi = nonlinsteerNoColAvoid(solver, current_state, current_goal_state, num_states, num_inputs, N, T, U,
                                                 X, P, DELTA, current_ref_traj, current_ref_inputs)

        # x_casadi, u_casadi = tracker.NonLinSteer(current_state, current_goal_state, current_ref_traj, current_ref_inputs)

        if x_casadi == []:
            print("nmpc failed at itr: ", itr)
            break
            # return [], []

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