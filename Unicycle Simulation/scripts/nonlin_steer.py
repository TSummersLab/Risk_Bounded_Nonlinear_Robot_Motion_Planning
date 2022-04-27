#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create steering function used as the basis for steering variations in the RRT* code and other scripts

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

Unicycle nonlinear exact steering code using Casadi SX. The code is used as a basis for its variants in RRT* and nmpc

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


import casadi as ca
from casadi import *
import casadi.tools as ca_tools
import numpy as np
import numpy.linalg as la
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import os
import numpy.random as npr

import file_version
FILEVERSION = file_version.FILEVERSION  # version of this file
SAVETIME = str(int(time.time()))  # Used in filename when saving data

SAVEPATH = os.path.join(os.path.curdir, 'saved_data')
try:  # try to create save directory if it doesn't exist
    os.mkdir(SAVEPATH)
except:
    print('Save directory exists')

class DrawTrajectory(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array, rob_diam=0.3,
                 export_fig=False):
        # Infer the input parameters
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0

        # Start a figure
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
        self.fig.set_size_inches(7, 6.5)
        # initialize the animation for plot
        self.animation_init()

        # Animate the figure
        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('./v1.gif', writer='imagemagick', fps=60)
            self.ani.save('v1.mp4', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # Plot target state as circle
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)

        # Plot the arrow in target state orientation with given orientation
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]),
                                         width=0.2, color='b')
        self.ax.add_patch(self.target_arr)

        # Plot robot state as circle
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)

        # Plot the arrow in robot state with given orientation
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]),
                                        width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)

        # return the handles for target state, robot state and arrow handles
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        # Infer the robot position and orientation from the state
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]

        # Set the inferred position as robot body center
        self.robot_body.center = position

        # Remove the previous iterate arrow
        self.robot_arr.remove()

        # Add new robot orientation arrow
        self.robot_arr = mpatches.Arrow(position[0], position[1],
                                        self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation),
                                        width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body

def sim_state(T, x0, u, f):
    f_value = f(x0, u)
    st = x0+T * f_value.T
    return st

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
    Q = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.001]])
    R = 100 * np.array([[1.0, 0.0],
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
    P = SX.sym('P', n_states + n_states)  # first and last states as independent parameters

    # Concatinate the decision variables (inputs and states)
    opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

    # Cost function
    obj = 0  # objective/cost
    g = []  # equality constraints
    g.append(X[0, :].T - P[:3])  # add constraint on initial state
    for i in range(N):
        # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
        # obj += 0*mtimes([(X[i, :]-P[3:].T), Q, (X[i, :]-P[3:].T).T])  # quadratic penalty on state deviation from target
        obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
        # obj += if_else(norm_1(X[i, :]-P[3:].T) > 5e-1, 10, 0)  # encourage minimum time

        # compute the next state from the dynamics
        x_next_ = f(X[i, :], U[i, :]) * T + X[i, :]

        # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
        g.append(X[i + 1, :].T - x_next_.T)

    g.append(X[N, :].T - P[3:])  # constraint on final state
    # obj = obj+mtimes([(X[N, :]-P[3:].T), Q, (X[N, :]-P[3:].T).T])  # final state cost

    # Set the nlp problem
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

    # Set the nlp problem settings
    # opts_setting = {'ipopt.max_iter': 1000,
    #                 'ipopt.print_level': 5,
    #                 'print_time': 0,
    #                 'verbose': 1,
    #                 'error_on_fail': 1,
    #                 'ipopt.tol': 1e-8,
    #                 'ipopt.acceptable_tol': 1e-8,
    #                 'ipopt.acceptable_obj_change_tol': 1e-8}
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

def nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx):
    """
    Solves the nonlinear steering problem using the solver from SetUpSteeringLawParameters
    Inputs:
        solver: Casadi NLP solver from SetUpSteeringLawParameters
        x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
        n_states, n_controls: number of states and controls
        N: horizon
        T: time step
        lbg, lbx, ubg, ubx:  lower and upper (l,u) state and input (x,g) bounds
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
    init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
    try:
        res = solver(x0=init_decision_vars, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
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

def main():
    # Define simulation parameters (user defined)
    animate_ok = True # animate or don't
    save_fig = False # save figure or don't
    show_path = True # show path figure
    T = 0.2  # sampling time [s]
    N = 80  # prediction horizon
    rob_diam = 0.3  # [m]
    v_max = 0.6  # maximum linear velocity (m/s)
    v_min = - v_max  # minimum linear velocity (m/s)
    omega_max = 0.125 * (2 * np.pi)  # maximum angular velocity (rad/s)
    omega_min = -omega_max  # minimum angular velocity (rad/s)
    x_max = 5 # maximum state in the horizontal direction
    x_min = -5 # minimum state in the horizontal direction
    y_max = 5 # maximum state in the vertical direction
    y_min = -5 # minimum state in the vertical direction
    theta_max = np.inf # maximum state in the theta direction
    theta_min = -np.inf # minimum state in the theta direction
    start_node = [2, 4, 1.0 * np.pi] # start state (x,y,theta)
    goal_node = [0, 0, np.pi] # goal state (x,y,theta)

    # Set up the Casadi solver
    t_timer_start = time.time()
    [solver, f, n_states, n_controls, lbx, ubx, lbg, ubg] = SetUpSteeringLawParameters(N, T, v_max, v_min,
                                                                                       omega_max, omega_min,
                                                                                       x_max, x_min,
                                                                                       y_max, y_min,
                                                                                       theta_max, theta_min)

    t_timer_end = time.time()
    t_setup_time = t_timer_end - t_timer_start
    print("Time to set up solver: ", t_setup_time)

    # Solve the optimization problem
    t_timer_start = time.time()
    x0 = np.array(start_node).reshape(-1, 1)  # reshaped initial state
    x0_copy = x0.copy()  # copy of initial state
    xT = np.array(goal_node).reshape(-1, 1)  # final state
    x_casadi, u_casadi = nonlinsteer(solver, x0, xT, n_states, n_controls, N, T, lbg, lbx, ubg, ubx)
    t_timer_end = time.time()
    t_solve_time = t_timer_end - t_timer_start
    print("Time to solve the problem: ", t_solve_time)

    # Simulate the system using the found controls
    x_sim = np.zeros([N + 1, 3, 1])
    x_sim[0, :, :] = x0_copy
    x_now = x0_copy.reshape(3, 1)
    ctr = 0
    for ut in u_casadi:
        ctr += 1
        x_now = sim_state(T, x_now, ut, f)
        x_sim[ctr, :, :] = x_now
    x_sim = x_sim.reshape(N + 1, 3)

    # Print the final error
    print('The final destination error is:', (np.linalg.norm(x_casadi[-1, :].reshape(3, 1) - xT)))

    # Show the Animation
    if animate_ok:
        AnimatePlot = DrawTrajectory(rob_diam=rob_diam,
                                     init_state=x0_copy,
                                     target_state=xT,
                                     robot_states=x_casadi)

    if show_path:
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.plot(x0_copy[0], x0_copy[1], 'D', color='yellow', markersize=10, markeredgecolor='blue')
        plt.plot(xT[0], xT[1], 'D', color='yellow', markersize=10, markeredgecolor='green')
        plt.plot(x_casadi[:, 0], x_casadi[:, 1], 'o', color='black')
        plt.plot(x_sim[:, 0], x_sim[:, 1], 'x', color='red')
        plt.plot(x_casadi[0, 0], x_casadi[0, 1], 'x', color='white')
        plt.plot(x_sim[0, 0], x_sim[0, 1], '+', color='magenta')
        plt.plot(x_casadi[-1, 0], x_casadi[-1, 1], '+', color='m')
        plt.plot(x_sim[-1, 0], x_sim[-1, 1], '+', color='blue')
        plt.legend(['start', 'goal', 'ipopt', 'sim', 'ipopt first', 'sim first', 'ipopt last', 'sim last'],
                   loc='lower right')

    if show_path and save_fig:
        plot_name = 'plot_path_' + FILEVERSION + '_' + SAVETIME + '.png'
        plot_name = os.path.join(SAVEPATH, plot_name)
        plt.savefig(plot_name)
    plt.show()

if __name__ == '__main__':
    main()