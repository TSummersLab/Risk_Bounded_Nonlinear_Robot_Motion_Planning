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

Tests nonlin_steer file to check radius-horizon compatibility using Monte Carlo

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

    MC = 1000 # number of Monte Carlo samples
    success_count = 0 # success rate counter
    goal = [0, 0, 0]
    starts = []

    r = 2 # radius of ball for goal_node about start_node
    # compute expected horizon length
    N_rotation = (2 * np.pi)/min(abs(omega_max)*T, abs(omega_min)*T)
    N_translation = r/min(abs(v_max)*T, abs(v_min)*T)
    N = int(ceil(N_rotation + N_translation)*1.1)

    t0 = time.time()
    for mc in range(MC):
        print(mc+1, '/', MC)
        start_node = [r * (2 * npr.rand() - 1), r * (2 * npr.rand() - 1), 2.0 * np.pi * npr.rand()]
        goal_node = [0, 0, 2.0 * np.pi * npr.rand()]

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

        if not x_casadi == []:
            success_count += 1
            starts.append(np.array([ start_node[0], start_node[1] ]))
    t1 = time.time()
    print('Success Percentage: ', success_count/MC)
    print('MC took: ', t1-t0, 'sec')
    print('Radius: ', r)
    print('N: ', N)

    # starts = np.array(starts).reshape(3,success_count)
    starts = np.array(starts).T
    x_starts = starts[0]
    y_starts = starts[1]
    plt.xlim(x_min-1, x_max+1)
    plt.ylim(y_min-1, y_max+1)

    plt.plot(x_starts, y_starts, 'o', color='black')
    plt.plot(goal[0], goal[1], 'o', color='red')
    plt.show()

if __name__ == '__main__':
    main()