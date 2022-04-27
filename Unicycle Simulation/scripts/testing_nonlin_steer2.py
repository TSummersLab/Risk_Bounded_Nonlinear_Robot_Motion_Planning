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

Wrapper for nonlin_steer where its functions are used to steer between points (automatically determines horizon in the
same way done in shorten_path)

Tested platform:
- Python 3.6.9 on Ubuntu 18.04 LTS (64 bit)

"""

import casadi as ca
from casadi import *
import casadi.tools as ca_tools
import numpy as np
import numpy.linalg as la
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy.random as npr

from nonlin_steer import nonlinsteer, SetUpSteeringLawParameters


def main():
    # Define simulation parameters (user defined)
    T = 0.2  # sampling time [s]
    v_max = 0.5  # maximum linear velocity (m/s)
    v_min = - v_max  # minimum linear velocity (m/s)
    omega_max = np.pi  # maximum angular velocity (rad/s)
    omega_min = -omega_max  # minimum angular velocity (rad/s)
    x_max = 5 # maximum state in the horizontal direction
    x_min = -5 # minimum state in the horizontal direction
    y_max = 5 # maximum state in the vertical direction
    y_min = -5 # minimum state in the vertical direction
    theta_max = np.inf # maximum state in the theta direction
    theta_min = -np.inf # minimum state in the theta direction

    start_node = np.array([-3., -4., 0.])
    goal_node = np.array([-2.7299038, -3.78969406, -0.69414571])
    # compute expected horizon length
    linear_distance = la.norm(start_node[0:2] - goal_node[0:2])
    angular_distance = abs(goal_node[2] - start_node[2])

    # method 1
    # N_translation = linear_distance / (v_max * T)
    # N_rotation = angular_distance / (omega_max * T)
    # N = int(ceil(N_rotation + N_translation) * 1.5)
    # method 2
    N_translation = linear_distance / (v_max/2 * T)
    N_rotation = angular_distance / (omega_max/2 * T)
    N = int(ceil(N_rotation + N_translation) * 1.1)


    # Set up the Casadi solver
    [solver, _, n_states, n_controls, lbx, ubx, lbg, ubg] = SetUpSteeringLawParameters(N, T, v_max, v_min,
                                                                                       omega_max, omega_min,
                                                                                       x_max, x_min,
                                                                                       y_max, y_min,
                                                                                       theta_max, theta_min)
    # Solve the optimization problem
    x0 = np.array(start_node).reshape(-1, 1)  # reshaped initial state
    xT = np.array(goal_node).reshape(-1, 1)  # final state
    x_casadi, u_casadi = nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx)

    if x_casadi == []:
        print('Failed with N = ', N)
    else:
        print('success with N = ', N)


if __name__ == '__main__':
    main()