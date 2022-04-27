# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:26:33 2021

@author: vxr131730
"""


import numpy as np
import numpy.random as npr
import time
import matplotlib.pyplot as plt

import os

from tracking_controller import dtime_dynamics, OpenLoopController, LQRController
from collision_check import PtObsColFlag, LineObsColFlag
from drrrts_nmpc import drrrtstar_with_nmpc

from utility.pickle_io import pickle_import, pickle_export
from opt_path import load_pickle_file
from utility.matrixmath import mdot

import copy

import config
STEER_TIME = config.STEER_TIME  # Maximum Steering Time Horizon
DT = config.DT  # timestep between controls
RANDAREA = config.RANDAREA
GOALAREA = config.GOALAREA  #[xmin,xmax,ymin,ymax]
VELMIN, VELMAX = config.VELMIN, config.VELMAX
ANGVELMIN, ANGVELMAX = config.ANGVELMIN, config.ANGVELMAX
OBSTACLELIST = config.OBSTACLELIST
ROBRAD = config.ROBRAD
SIGMAW = config.SIGMAW # TODO - do *10^5, *10^4, *1 - different noise levels
QLL = config.QLL
RLL = config.RLL
QTLL = config.QTLL

SAVEPATH = config.SAVEPATH

# TODO change this as needed
# 0: experimental
# 1: I think Env1 w/ DR + nrm noise NOTE: nmpc is drnmpc and nmpc with no dr is not there
# 3: Env3 w/ DR + nrm noise
# 4: Env3 w/ DR + lap noise
# 5: Env3 w/ DR + gum noise
# 10: experimental
# 100--103: Testing out final stuff
# 200--299: IROS data: laplace

MC_FOLDER = os.path.join('..', 'monte_carlo', 'env000400')
std_dev = 0.003 # 

# MC_FOLDER = os.path.join('..', 'monte_carlo', 'env000200')
# std_dev = 0.0000005

PROBLEM_DATA_STR = 'problem_data'
RESULT_DATA_STR = 'result_data'
PICKLE_EXTENSION = '.pkl'


def load_ref_traj(input_file):
    inputs_string = "inputs"
    states_file = input_file.replace(inputs_string, "states")

    input_file = os.path.join(SAVEPATH, input_file)
    states_file = os.path.join(SAVEPATH, states_file)

    # load inputs and states
    ref_inputs = load_pickle_file(input_file)
    ref_states = load_pickle_file(states_file)
    return ref_states, ref_inputs


def idx2str(idx):
    # This function establishes a common data directory naming specification
    return '%012d' % idx


def import_problem_data(idx, mc_folder=None):
    if mc_folder is None:
        mc_folder = MC_FOLDER
    path_in = os.path.join(mc_folder, idx2str(idx), PROBLEM_DATA_STR+PICKLE_EXTENSION)
    problem_data = pickle_import(path_in)
    return problem_data


def export_problem_data(problem_data, idx, mc_folder=None):
    if mc_folder is None:
        mc_folder = MC_FOLDER
    dirname_out = os.path.join(mc_folder, idx2str(idx))
    filename_out = PROBLEM_DATA_STR+PICKLE_EXTENSION
    pickle_export(dirname_out, filename_out, problem_data)
    return


def import_result_data(idx, controller_str, mc_folder=None):
    if mc_folder is None:
        mc_folder = MC_FOLDER
    path_in = os.path.join(mc_folder, idx2str(idx), RESULT_DATA_STR+'_'+controller_str+PICKLE_EXTENSION)
    result = pickle_import(path_in)
    return result


def export_result_data(problem_data, idx, controller_str, mc_folder=None):
    if mc_folder is None:
        mc_folder = MC_FOLDER
    dirname_out = os.path.join(mc_folder, idx2str(idx))
    filename_out = RESULT_DATA_STR+'_'+controller_str+PICKLE_EXTENSION
    pickle_export(dirname_out, filename_out, problem_data)
    return


def generate_disturbance_history(common_data, seed=None, dist=None, show_hist=False):
    rng = npr.default_rng(seed)
    sigma1 = SIGMAW[0, 0]  # first entry in SigmaW
    x_ref_hist = common_data['x_ref_hist']
    T = x_ref_hist.shape[0]

    if dist is None:
        dist = "nrm"  # "nrm", "lap", "gum"

    if dist == "nrm":
        w_hist = rng.multivariate_normal(mean=[0, 0, 0], cov=SIGMAW, size=T)
    elif dist == "lap":
        l = 0
        b = (sigma1 / 2) ** 0.5
        w_hist = rng.laplace(loc=l, scale=b, size=[T, 3])  # mean = loc, var = 2*scale^2
    elif dist == "gum":
        b = (6*sigma1)**0.5/np.pi
        l = -0.57721*b
        w_hist = rng.gumbel(loc=l, scale=b, size=[T, 3])  # mean = loc+0.57721*scale, var = pi^2/6 scale^2
    else:
        raise ValueError('Invalid disturbance generation method!')

    if show_hist:
        plt.hist(w_hist)
        plt.show()
    return w_hist


def make_idx_list(num_trials, offset=0):
    idx_list = []
    for i in range(num_trials):
        idx = i+offset+1
        idx_list.append(idx)
    return idx_list


def make_problem_data(T, num_trials, offset=0, dist='nrm'):
    idx_list = []
    for i in range(num_trials):
        idx = i + offset + 1
        w_hist = generate_disturbance_history(T, seed=idx, dist=dist) # Process Noise
        v_hist = generate_disturbance_history(T, seed=idx, dist=dist) # Sensor Noise
        problem_data = {'w_hist': w_hist, 'v_hist': v_hist}
        export_problem_data(problem_data, idx)
        idx_list.append(idx)
    return idx_list


class nmpc_controller_object(): # place holder for nmpc controller object
    def __init__(self):
        self.name = 'nmpc'


class drnmpc_controller_object(): # place holder for drnmpc controller object
    def __init__(self):
        self.name = 'drnmpc'


class hnmpc_controller_object(): # place holder for hnmpc controller object
    def __init__(self):
        self.name = 'hnmpc'


def make_controller(controller_str, x_ref_hist, u_ref_hist):
    """
    Creates the controller object and returns the time it took to do so
    """
    from tracking_controller import create_lqrm_controller

    if controller_str == 'open-loop':
        # Create open-loop controller object
        time_start = time.time()
        controller = OpenLoopController(u_ref_hist)
        time_stop = time.time()
    elif controller_str == 'lqr':
        # Create vanilla LQR controller object
        time_start = time.time()
        controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=False)
        time_stop = time.time()
    elif controller_str == 'lqrm':
        # Create robust LQR controller object
        time_start = time.time()
        controller = create_lqrm_controller(x_ref_hist, u_ref_hist, use_robust_lqr=True)
        time_stop = time.time()
    elif controller_str == 'nmpc':
        # Create nmpc controller object
        time_start = time.time()
        controller = nmpc_controller_object()
        time_stop = time.time()
    elif controller_str == 'drnmpc':
        # Create dr-nmpc controller object
        time_start = time.time()
        controller = drnmpc_controller_object()
        time_stop = time.time()
    elif controller_str == 'hnmpc':
        # Create dr-nmpc controller object
        time_start = time.time()
        controller = hnmpc_controller_object()
        time_stop = time.time()
    else:
        raise ValueError('Invalid controller string')

    controller.name = controller_str
    return [controller, time_stop - time_start]


def rollout(n, m, T, DT, x0=None, w_hist=None, controller=None, saturate_inputs=True):
    # Initialize
    x_hist = np.zeros([T, n])
    if x0 is None:
        x0 = np.zeros(n)
    x_hist[0] = x0
    x = np.copy(x0)
    u_hist = np.zeros([T, m])

    collision_flag = False
    collision_idx = None

    # Simulate
    for t in range(T-1):
        # Compute desired control inputs
        u = controller.compute_input(x, t)
        # Saturate inputs at actuator limits
        if saturate_inputs:
            u[0] = np.clip(u[0], VELMIN, VELMAX)
            u[1] = np.clip(u[1], ANGVELMIN, ANGVELMAX)
        # Get disturbance
        w = w_hist[t]
        # Transition the state
        x_old = np.copy(x)
        x = dtime_dynamics(x, u, DT) + w
        # Check for collision
        if PtObsColFlag(x, OBSTACLELIST, RANDAREA, ROBRAD) or LineObsColFlag(x_old, x, OBSTACLELIST, ROBRAD):
            collision_flag = True
            collision_idx = t
            x_hist[t+1:] = x  # pad out x_hist with the post-collision state
            break
        # Record quantities
        x_hist[t+1] = x
        u_hist[t] = u

    result_data = {'x_hist': x_hist,
                   'u_hist': u_hist,
                   'collision_flag': collision_flag,
                   'collision_idx': collision_idx}
    return result_data


def trial(problem_data, common_data, controller, setup_time):
    # Get the reference trajectory and controller
    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']

    # Number of states, inputs
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape

    # Start in the reference initial state
    x0 = x_ref_hist[0]

    # Get the disturbance for this trial
    w_hist = problem_data['w_hist']
    v_hist = problem_data['v_hist']

    # Simulate trajectory with noise and control, forwards in time
    if type(controller) in [OpenLoopController, LQRController]:
        time_start = time.time()
        result_data = rollout(n, m, T, DT, x0, w_hist, controller=controller)
        time_stop = time.time()
    elif controller.name == 'hnmpc':
        time_start = time.time()
        nmpc_horizon = 10
        result_data = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=w_hist, v=v_hist, drnmpc = True, hnmpc = True)
        time_stop = time.time()
    elif controller.name == 'drnmpc':
        time_start = time.time()
        nmpc_horizon = 10
        result_data = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=w_hist, v=v_hist, drnmpc = True, hnmpc = False)
        time_stop = time.time()
    elif controller.name == 'nmpc':
        time_start = time.time()
        nmpc_horizon = 10
        result_data = drrrtstar_with_nmpc(nmpc_horizon, x_ref_hist, u_ref_hist, n, m, T, w=w_hist, v=v_hist, drnmpc=False, hnmpc = False)
        time_stop = time.time()
    else:
        raise NotImplementedError('Need rollout function for NMPC!')

    try:  # try to add setup time if run time exists
        result_data["run_time"] += setup_time
    except:  # if run time didn't exist, create it and set it's value to the setup time + execution time
        result_data["run_time"] = setup_time + time_stop - time_start

    return result_data


def monte_carlo(idx_list, common_data, controller_list, setup_time_list, verbose=False):
    num_trials = len(idx_list)
    for i, idx in enumerate(idx_list):
        if verbose:
            print('Trial %6d / %d    ' % (i+1, num_trials), end='')
            print('Problem %s    ' % idx2str(idx), end='')
        problem_data = import_problem_data(idx)

        if verbose:
            print('Simulating...', end='')
        for j, controller in enumerate(controller_list):
            result_data = trial(problem_data, common_data, controller, setup_time_list[j])
            export_result_data(result_data, idx, controller.name)
        if verbose:
            print(' complete.')
    return


def aggregate_results(idx_list, controller_str_list, mc_folder=None):
    if mc_folder is None:
        mc_folder = MC_FOLDER
    result_data_dict = {}
    for controller_str in controller_str_list:
        result_data_list = []
        for idx in idx_list:
            result_data = import_result_data(idx, controller_str, mc_folder)
            result_data_list.append(result_data)
        result_data_dict[controller_str] = result_data_list
    return result_data_dict


def cost_of_trajectory(x_ref_hist, x_hist, u_hist):
    T = x_hist.shape[0]
    dxtot = 0
    utot = 0
    for t in range(T):
        if t < T-1:
            Q = QLL
        else:
            Q = QTLL
        R = RLL
        dx = x_hist[t] - x_ref_hist[t]
        u = u_hist[t]
        dxtot += mdot(dx.T, Q, dx)
        utot += mdot(u.T, R, u)
    return dxtot + utot


def score_trajectory(x_ref_hist, u_ref_hist, x_hist, u_hist):
    # NOTE: x_hist should be collision-free
    ref_cost = cost_of_trajectory(x_ref_hist, x_ref_hist, u_ref_hist)
    cost = cost_of_trajectory(x_ref_hist, x_hist, u_hist)
    score = cost/ref_cost
    return score


def metric_trials(result_data_list, common_data, skip_scores=False):
    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']
    N = len(result_data_list)
    num_collisions = 0
    collisions = np.full(N, False)
    score_sum = 0.0
    scores = np.zeros(N)
    run_time_sum = 0.0
    nlp_fail = np.full(N, False)
    for i, result_data in enumerate(result_data_list):
        x_hist = result_data['x_hist']
        u_hist = result_data['u_hist']
        collision_flag = result_data['collision_flag']
        run_time = result_data['run_time']
        run_time_sum += run_time
        try:
            nlp_fail_flag = result_data["nlp_failed_flag"]
            if nlp_fail_flag:
                nlp_fail[i] = True
        except :
            pass
        if collision_flag:
            num_collisions += 1
            collisions[i] = True
            scores[i] = np.inf
        else:
            if skip_scores:
                score = -1
            else:
                score = score_trajectory(x_ref_hist, u_ref_hist, x_hist, u_hist)
            scores[i] = score
            score_sum += score
    run_time_avg = run_time_sum/N
    score_avg = score_sum/N
    collision_avg = num_collisions/N
    return scores, score_avg, collisions, collision_avg, run_time_avg, nlp_fail


def metric_controllers(result_data_dict, common_data, skip_scores=False):
    metric_dict = {}
    for controller_str, result_data_list in result_data_dict.items():
        scores, score_avg, collisions, collision_avg, run_time_avg, nlp_fail = metric_trials(result_data_list, common_data, skip_scores)
        metric_dict[controller_str] = {'scores': scores,
                                       'score_avg': score_avg,
                                       'collisions': collisions,
                                       'collision_avg': collision_avg,
                                       'run_time_avg': run_time_avg,
                                       'nlp_fail': nlp_fail}
    return metric_dict


def score_histogram(score_dict):
    fig, ax = plt.subplots()
    for controller_str, c_score_dict in score_dict.items():
        scores = c_score_dict['scores']
        num_bins = 40
        ax.hist(scores[np.isfinite(scores)], bins=num_bins, density=True, alpha=0.5, label=controller_str)
    ax.legend()
    return fig, ax


def plotter(result_data_dict, common_data):
    # Plotting
    from plotting import plot_paths

    x_ref_hist = common_data['x_ref_hist']
    u_ref_hist = common_data['u_ref_hist']
    T, n = x_ref_hist.shape
    T, m = u_ref_hist.shape
    t_hist = np.arange(T) * DT

    x_hist_all_dict = {}
    collision_flag_all_dict = {}

    for controller_str, result_data_list in result_data_dict.items():
        x_hist_all = np.array([result_data['x_hist'] for result_data in result_data_list])
        collision_flag_all = [result_data['collision_flag'] for result_data in result_data_list]
        x_hist_all_dict[controller_str] = x_hist_all
        collision_flag_all_dict[controller_str] = collision_flag_all

    # Plot Monte Carlo paths
    ax_lim = [-5.2, 5.2, -5.2, 5.2]
    fig, ax = plot_paths(t_hist, x_hist_all_dict, collision_flag_all_dict, x_ref_hist, title=None, fig_offset=None,  axis_limits=ax_lim)
    return fig, ax


if __name__ == "__main__":
    plt.close('all')

    input_file = 'OptTraj_short_v1_0_1625163893_inputs'

    x_ref_hist, u_ref_hist = load_ref_traj(input_file)

    controller_str_list = ['nmpc']
    
    controller_objects_and_init_time = [make_controller(controller_str, x_ref_hist, u_ref_hist) for controller_str in controller_str_list]
    controller_list = [result[0] for result in controller_objects_and_init_time] # extract controller list
    setup_time_list = [result[1] for result in controller_objects_and_init_time] # extract time to create controller object
    # controller_list, time_list = [make_controller(controller_str, x_ref_hist, u_ref_hist) for controller_str in controller_str_list]
    common_data = {'x_ref_hist': x_ref_hist,
                   'u_ref_hist': u_ref_hist}

    num_trials = 1000
    trials_offset = 0

    std_dev_str = '%.7f' % std_dev
    std_dev_str = std_dev_str.replace('.', 'p')

    # Set this true to run new Monte Carlo trials, set to false to pull in saved data
    run_flag = False
    if run_flag:
        # Make new problem data
        idx_list = make_problem_data(common_data, num_trials=num_trials, offset=trials_offset, dist='nrm')
        # Run the monte carlo simulation
        monte_carlo(idx_list, common_data, controller_list, setup_time_list, verbose=True)
    else:
        idx_list = make_idx_list(num_trials, offset=trials_offset)

    # # Plot all controllers together
    # result_data_dict = aggregate_results(idx_list, controller_str_list)
    # plotter(result_data_dict, common_data)
    # # Metrics
    # metric_dict = metric_controllers(result_data_dict, common_data)

    # Plot each controller separately
    for controller_str in controller_str_list:
        my_list = [controller_str]
        result_data_dict = aggregate_results(idx_list, my_list)
        fig, ax = plotter(result_data_dict, common_data)

        dirname_out = os.path.join('..', 'monte_carlo', 'path_plots')
        filename_out = 'path_plot_'+controller_str+'_gauss_1'+'.png'
        path_out = os.path.join(dirname_out, filename_out)
        fig.savefig(path_out, dpi=600)

        # Metrics
        metric_dict = metric_controllers(result_data_dict, common_data)

    # print('lqrm total time: ', [result['run_time'] for result in result_data_dict['lqrm']])
    # print('nmpc total time: ', [result['run_time'] for result in result_data_dict['nmpc']])
    # print('nmpc NLP failed: ', [result['nlp_failed_flag'] for result in result_data_dict['nmpc']])
    # print('drnmpc NLP failed: ', [result['nlp_failed_flag'] for result in result_data_dict['drnmpc']])
    for controller_str, c_metric_dict in metric_dict.items():
        collisions = c_metric_dict['collisions']
        print('%s failed %d / %d ' % (controller_str, int(np.sum(collisions)), num_trials))
    for controller_str, c_metric_dict in metric_dict.items():
        avg_score = c_metric_dict['score_avg']
        print('%s score average is %f ' % (controller_str, avg_score))
    for controller_str, c_metric_dict in metric_dict.items():
        run_time_avg = c_metric_dict['run_time_avg']
        print('%s average run time is %f ' % (controller_str, run_time_avg))
    for controller_str, c_metric_dict in metric_dict.items():
        if controller_str == 'hnmpc' or controller_str == 'drnmpc' or controller_str == 'nmpc':
            nlp_fail = c_metric_dict['nlp_fail']
            print('%s nlp failed %d / %d ' % (controller_str, int(np.sum(nlp_fail)), num_trials))
