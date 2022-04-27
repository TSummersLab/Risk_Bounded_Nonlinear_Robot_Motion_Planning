#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Ben Gravell
Email:
benjamin.gravell@utdallas.edu
Github:
@BenGravell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting and animation functions used elsewhere in the code

Tested platform:
- TODO: add platform

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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
import matplotlib.animation as ani
import copy
import os

import config
ROBRAD = config.ROBRAD
OBSTACLELIST = copy.copy(config.OBSTACLELIST)
RANDAREA = config.RANDAREA
GOALAREA = config.GOALAREA


def convert_controller_str_nice(old_str):
    if old_str == 'open-loop':
        return 'Open-loop'
    elif old_str == 'lqr':
        return 'LQR'
    elif old_str == 'lqrm':
        return 'Robust LQR'
    elif old_str == 'nmpc':
        return 'NMPC'


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def make_plot_properties(quantity):
    colors = ['k', 'tab:orange', 'tab:blue']
    styles = ['--', '-', '-']
    widths = [4, 1, 1]
    alphas = [0.5, 0.8, 1.0]
    labels = ['Reference', 'Open-loop', 'Closed-loop']
    if quantity == 'state' or quantity == 'disturbance':
        ylabels = ['x', 'y', 'theta (deg)']
    elif quantity == 'input':
        ylabels = ['Speed', 'Steering Rate']
    elif quantity == 'disturbance_base':
        ylabels = ['rolling', 'transverse', 'theta (deg)']
    return colors, styles, widths, alphas, labels, ylabels


def plot_hist(t_hist, hist_list, quantity, figsize=(8, 8)):
    n = hist_list[0].shape[1]
    fig, ax = plt.subplots(nrows=n, sharex=True, figsize=figsize)
    colors, styles, widths, alphas, labels, ylabels = make_plot_properties(quantity)
    for i in range(n):
        ylabel = ylabels[i]
        for hist, color, style, width, alpha in zip(hist_list, colors, styles, widths, alphas):
            x_plot = np.copy(hist[:, i])
            if quantity == 'state':
                if i == 2:
                    x_plot *= 360/(2*np.pi)
            ax[i].plot(t_hist, x_plot, color=color, linestyle=style, lw=width, alpha=alpha)
        ax[i].legend(labels)
        ax[i].set_ylabel(ylabel)
    ax[-1].set_xlabel('Time')
    ax[0].set_title(quantity)
    fig.tight_layout()
    return fig, ax


def plot_gain_hist(K_hist):
    n, m = K_hist.shape[2], K_hist.shape[1]
    fig, ax = plt.subplots()
    for i in range(m):
        for j in range(n):
            ax.plot(K_hist[:, i, j], label='(%1d, %1d)' % (i, j))
    ax.legend()
    ax.set_title('Entrywise Gains (K)')
    return fig, ax


def animate(t_hist, x_hist, u_hist, x_ref_hist=None, u_ref_hist=None, show_start=True, show_goal=True,
            title=None, repeat=False, fig_offset=None, axis_limits=None,
            robot_w=1.2, robot_h=2.0, wheel_w=2.0, wheel_h=0.50):
    if axis_limits is None:
        axis_limits = [-20, 20, -10, 30]
    T = t_hist.size

    def draw_robot(x, ax, pc_list_robot=None,
                   body_width=robot_w, body_height=robot_h, wheel_width=wheel_w, wheel_height=wheel_h):
        if pc_list_robot is not None:
            for pc in pc_list_robot:
                try:
                    ax.collections.remove(pc)
                except:
                    pass
        cx, cy, angle = x[0], x[1], x[2]*360/(2*np.pi)
        # TODO - just apply transforms to existing patches instead of redrawing patches every frame
        transform = Affine2D().rotate_deg_around(cx, cy, angle)
        patch_body = Rectangle((cx-body_width/2, cy-body_height/2), body_width, body_height)
        patch_body.set_transform(transform)
        pc_body = PatchCollection([patch_body], color=[0.7, 0.7, 0.7], alpha=0.8)
        patch_wheel1 = Rectangle((cx-wheel_width/2, cy-body_height/2-wheel_height), wheel_width, wheel_height)
        patch_wheel2 = Rectangle((cx-wheel_width/2, cy+body_height/2), wheel_width, wheel_height)
        patch_wheel1.set_transform(transform)
        patch_wheel2.set_transform(transform)
        pc_wheel = PatchCollection([patch_wheel1, patch_wheel2], color=[0.3, 0.3, 0.3], alpha=0.8)
        pc_list_robot = [pc_body, pc_wheel]
        for pc in pc_list_robot:
            ax.add_collection(pc)
        return pc_list_robot

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if fig_offset is not None:
        move_figure(fig, fig_offset[0], fig_offset[1])

    # Draw the realized path
    ax.plot(x_hist[:, 0], x_hist[:, 1],
            color='tab:blue', linestyle='-', lw=3, alpha=0.8, label='Realized path', zorder=2)

    # Draw the reference path and start/goal locations
    if x_ref_hist is not None:
        ax.plot(x_ref_hist[:, 0], x_ref_hist[:, 1],
                color='k', linestyle='--', lw=3, alpha=0.5, label='Reference path', zorder=1)
        if show_start:
            ax.scatter(x_ref_hist[0, 0], x_ref_hist[0, 1], s=200, c='tab:blue', marker='o', label='Start', zorder=10)
        if show_goal:
            ax.scatter(x_ref_hist[-1, 0], x_ref_hist[-1, 1], s=300, c='goldenrod', marker='*', label='Goal', zorder=20)
        plot_env(ax)

    # Draw the robot for the first time
    pc_list_robot = draw_robot(x_hist[0], ax)

    # Initialize plot options
    ax.axis('equal')
    ax.axis(axis_limits)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.autoscale(False)
    ax.set_title(title)
    ax.legend(labelspacing=1)
    fig.tight_layout()

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.legend().get_texts()):
        item.set_fontsize(20)

    def update(i, pc_list_robot):
        pc_list_robot = draw_robot(x_hist[i], ax, pc_list_robot)
        return pc_list_robot

    anim = ani.FuncAnimation(fig, update, fargs=[pc_list_robot], frames=T-1, interval=1, blit=True, repeat=repeat) # increase interval to make animation slower
    plt.show()
    return fig


def plot_env(ax):
    # Plot the goal
    x1mingoal = GOALAREA[0]
    x1maxgoal = GOALAREA[1]
    y1mingoal = GOALAREA[2]
    y1maxgoal = GOALAREA[3]
    goalHeight = y1mingoal
    xGoal = np.arange(x1mingoal, x1maxgoal, 0.2).tolist()  # [-5.0,-4.8, -4.6, -4.4, -4.2, -4.0]
    y1Goal = [goalHeight]

    # Shade the area between y1 and line y=y1maxgoal
    # ax = plt.axes()
    # goal_color = '#5fe0b7'
    goal_color = 'tab:green'
    ax.fill_between(xGoal, y1Goal, y1maxgoal,
                    facecolor='#5fe0b7',
                    edgecolor='#154734',
                    linestyle='--',
                    linewidth=4,
                    alpha=0.6, zorder=100)

    # Plot the padded obstacles
    color = 'k'
    obstacles = [Rectangle(xy=(ox-ROBRAD, oy-ROBRAD),
                           width=wd+2 * ROBRAD,
                           height=ht+2 * ROBRAD,
                           angle=0,
                           color=color) for (ox, oy, wd, ht) in OBSTACLELIST]
    # Plot the environment boundary obstacles
    env_xmin, env_xmax, env_ymin, env_ymax = RANDAREA
    env_width = env_xmax-env_xmin
    env_height = env_ymax-env_ymin
    env_thickness = 0.1
    env_obstacles = [Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                               width=env_thickness+ROBRAD,
                               height=env_height+2 * env_thickness,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmax-ROBRAD, env_ymin-env_thickness),
                               width=env_thickness+ROBRAD,
                               height=env_height+2 * env_thickness,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                               width=env_width+2 * env_thickness,
                               height=env_thickness+ROBRAD,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmin-env_thickness, env_ymax-ROBRAD),
                               width=env_width+2 * env_thickness,
                               height=env_thickness+ROBRAD,
                               angle=0, color=color)
                     ]
    for obstacle in env_obstacles:
        ax.add_artist(obstacle)
    for obstacle in obstacles:
        ax.add_artist(obstacle)

    # Plot the obstacles
    color = 'k'
    # color = (0.1, 0.1, 0.1)
    obstacles = [Rectangle(xy=(ox, oy),
                           width=wd,
                           height=ht,
                           angle=0,
                           color=color) for (ox, oy, wd, ht) in OBSTACLELIST]
    # Plot the environment boundary obstacles
    env_obstacles = [Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                               width=env_thickness,
                               height=env_height+2 * env_thickness,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmax, env_ymin-env_thickness),
                               width=env_thickness,
                               height=env_height+2 * env_thickness,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                               width=env_width+2 * env_thickness,
                               height=env_thickness,
                               angle=0, color=color),
                     Rectangle(xy=(env_xmin-env_thickness, env_ymax),
                               width=env_width+2 * env_thickness,
                               height=env_thickness,
                               angle=0, color=color)
                     ]
    for obstacle in env_obstacles:
        ax.add_artist(obstacle)
    for obstacle in obstacles:
        ax.add_artist(obstacle)

    # Show the corner points of the environment
    # ax.scatter([env_xmin, env_xmin, env_xmax, env_xmax],
    #            [env_ymin, env_ymax, env_ymin, env_ymax],
    #            color='y', marker='d', s=30, zorder=10000)
    return ax


def plot_paths(t_hist, x_hist_all_dict, collision_flag_all_dict, x_ref_hist, show_start=True, show_goal=True,
               show_x_ref=True, title=None, fig_offset=None, axis_limits=None, show_legend=False):
        if axis_limits is None:
            axis_limits = [-1, 1, -1, 1]
        T = t_hist.size

        # Initialize plot
        fig, ax = plt.subplots(figsize=(3, 3))
        if fig_offset is not None:
            move_figure(fig, fig_offset[0], fig_offset[1])

        # Get collision points based on collision flags
        x_fail_points_dict = {}
        x_success_points_dict = {}
        for name, x_hist_all in x_hist_all_dict.items():
            success_point_list = []
            fail_point_list = []
            for i, x_hist in enumerate(x_hist_all):
                flag = collision_flag_all_dict[name][i]
                if flag:
                    fail_point_list.append(x_hist[-1])
                else:
                    success_point_list.append(x_hist[-1])

            x_fail_points_dict[name] = np.array(fail_point_list)
            x_success_points_dict[name] = np.array(success_point_list)

        # Draw the realized paths
        realized_path_colors = {'hnmpc': '#c13f21',
                                'nmpc': '#0078f0',
                                'drnmpc': '#dda032',
                                'lqr': '#3a9476',
                                'lqrm': '#810f7c',
                                'open-loop': '#e87500'}
        for name, x_hist_all in x_hist_all_dict.items():
            N = x_hist_all.shape[0]
            color = realized_path_colors[name]
            plt.plot([], label=name, color=color, alpha=0.8, lw=3)
            ax.plot(x_hist_all[:, :, 0].T,
                    x_hist_all[:, :, 1].T,
                    color=color, linestyle='-', lw=1, alpha=min(2.0/(N**0.5), 1.0), zorder=2)
        # Draw markers at the collision points to visualize failures
        for name, x_fail_points in x_fail_points_dict.items():
            color = realized_path_colors[name]
            if len(x_fail_points) > 0:
                ax.scatter(x_fail_points[:, 0],
                           x_fail_points[:, 1],
                           marker='x', color=color, zorder=100)
        for name, x_success_points in x_success_points_dict.items():
            color = realized_path_colors[name]
            if len(x_success_points) > 0:
                ax.scatter(x_success_points[:, 0],
                           x_success_points[:, 1],
                           marker='o', facecolors='none', edgecolors=color, zorder=100)

        # Draw the reference path and start/goal locations
        if x_ref_hist is not None:
            if show_x_ref:
                ax.plot(x_ref_hist[:, 0], x_ref_hist[:, 1],
                        color='k', linestyle='--', lw=2, alpha=0.8, label='reference', zorder=150)
            if show_start:
                ax.scatter(x_ref_hist[0, 0], x_ref_hist[0, 1], s=200, c='w', edgecolor='k', linewidths=2, marker='^', label='Start', zorder=200)
            if show_goal:
                ax.scatter(x_ref_hist[-1, 0], x_ref_hist[-1, 1], s=400, c='w', edgecolor='k', linewidths=2, marker='*', label='Goal', zorder=200)
            plot_env(ax)

        # Initialize plot options
        ax.axis('equal')
        ax.axis(axis_limits)
        ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.autoscale(False)
        ax.set_title(title)
        fontsize = 16
        if show_legend:
            ax.legend(ncol=2, prop={'size': fontsize}, bbox_to_anchor=(0.35, 0.25))

        # Change the font sizes
        font_items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        # if show_legend:
        #     font_items += ax.legend().get_texts()
        for item in font_items:
            item.set_fontsize(fontsize)

        fig.tight_layout()
        plt.show()
        return fig, ax


def get_collision_counts():
    from monte_carlo import load_ref_traj, make_idx_list, aggregate_results, metric_controllers

    input_file = 'OptTraj_short_v1_0_1614838308_inputs'
    x_ref_hist, u_ref_hist = load_ref_traj(input_file)
    common_data = {'x_ref_hist': x_ref_hist,
                   'u_ref_hist': u_ref_hist}

    sigmaw_vals = [0.0000005,
                   0.0000010,
                   0.0000050,
                   0.0000100,
                   0.0000500,
                   0.0001000,
                   0.0005000,
                   0.0010000,
                   0.0015000,
                   0.0020000,
                   0.0025000,
                   0.0030000,
                   0.0035000,
                   0.0040000,
                   0.0045000,
                   0.0050000,
                   0.0055000,
                   0.0060000,
                   0.0065000,
                   0.0070000,
                   0.0075000,
                   0.0080000,
                   0.0085000,
                   0.0090000,
                   0.0095000,
                   0.0100000,
                   0.0500000,
                   0.1000000]
    mc_env_strs = ['env000200',
                   'env000201',
                   'env000202',
                   'env000203',
                   'env000204',
                   'env000205',
                   'env000206',
                   'env000207',
                   'env000271',
                   'env000272',
                   'env000273',
                   'env000274',
                   'env000275',
                   'env000276',
                   'env000277',
                   'env000208',
                   'env000281',
                   'env000282',
                   'env000283',
                   'env000284',
                   'env000285',
                   'env000286',
                   'env000287',
                   'env000288',
                   'env000289',
                   'env000209',
                   'env000210',
                   'env000211']

    mc_env_folder = os.path.join('..', 'monte_carlo')

    controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']
    num_trials = 1000
    trials_offset = 0
    idx_list = make_idx_list(num_trials, offset=trials_offset)

    collision_count_dict = {}
    for controller_str in controller_str_list:
        collision_count_dict[controller_str] = []

    for mc_env_str in mc_env_strs:
        mc_folder = os.path.join(mc_env_folder, mc_env_str)
        result_data_dict = aggregate_results(idx_list, controller_str_list, mc_folder)

        # Metrics
        metric_dict = metric_controllers(result_data_dict, common_data, skip_scores=True)
        for controller_str in controller_str_list:
            count = np.sum(metric_dict[controller_str]['collisions'])
            collision_count_dict[controller_str].append(count)
    return sigmaw_vals, collision_count_dict, num_trials


def collision_stat_plot(stat_plot_data, collision_bound):
    sigmaw_vals = stat_plot_data['sigmaw_vals']
    collision_count_dict = stat_plot_data['collision_count_dict']
    num_trials = stat_plot_data['num_trials']

    plt.style.use('conlab.mplstyle')
    fig, ax = plt.subplots(figsize=(7, 4))
    marker_dict = {'open-loop': 's',
                   'lqr': '^',
                   'lqrm': '*',
                   'nmpc': 'o'}
    for controller_str, collision_count_vals in collision_count_dict.items():
        ax.semilogx(sigmaw_vals, collision_count_vals,
                    marker=marker_dict[controller_str],
                    markersize=10,
                    label=convert_controller_str_nice(controller_str))
    ax.axhline(collision_bound*num_trials, linestyle='--', color='tab:grey',
               label='Bound (%d%%)' % (100*collision_bound))
    ax.set_xlabel(r'$\sigma_w^2$')
    ax.set_ylabel('Number of collisions')
    ax.legend()
    fig.tight_layout()

    # Print stats
    for controller_str, collision_count_vals in collision_count_dict.items():
        yb = collision_bound*num_trials
        idx = np.max(np.where(np.array(collision_count_vals) < yb))
        # Conservative values
        print('%12s    %f' % (controller_str, sigmaw_vals[idx]))

    print('')
    for controller_str, collision_count_vals in collision_count_dict.items():
        yb = collision_bound * num_trials
        idx = np.max(np.where(np.array(collision_count_vals) < yb))
        # Interpolated values
        x1 = sigmaw_vals[idx]
        x2 = sigmaw_vals[idx+1]
        y1 = collision_count_vals[idx]
        y2 = collision_count_vals[idx+1]
        xb = x1 + ((yb-y1)/(y2-y1))*(x2-x1)
        print('%12s    %f' % (controller_str, xb))

    return fig, ax


def time_stats():
    from monte_carlo import load_ref_traj, make_idx_list, aggregate_results, metric_controllers

    input_file = 'OptTraj_short_v1_0_1614838308_inputs'
    x_ref_hist, u_ref_hist = load_ref_traj(input_file)
    common_data = {'x_ref_hist': x_ref_hist,
                   'u_ref_hist': u_ref_hist}
    mc_env_folder = os.path.join('..', 'monte_carlo')
    mc_env_str = 'env000220'

    controller_str_list = ['open-loop', 'lqr', 'lqrm', 'nmpc']
    num_trials = 120
    trials_offset = 0
    idx_list = make_idx_list(num_trials, offset=trials_offset)

    mc_folder = os.path.join(mc_env_folder, mc_env_str)
    result_data_dict = aggregate_results(idx_list, controller_str_list, mc_folder)

    # Metrics
    metric_dict = metric_controllers(result_data_dict, common_data, skip_scores=True)

    for controller_str in controller_str_list:
        t = metric_dict[controller_str]['run_time_avg']
        print('%s    %f' % (controller_str, t))


if __name__ == "__main__":
    # This code block does the stat plot & timing metrics
    plt.close('all')
    from utility.pickle_io import pickle_import, pickle_export

    dirname_out = os.path.join('..', 'monte_carlo', 'stat_plot_agg_data')
    filename_out = 'stat_plot_data.pkl'
    # # Aggregate data from scratch
    # sigmaw_vals, collision_count_dict, num_trials = get_collision_counts()
    # stat_plot_data = {'sigmaw_vals': sigmaw_vals,
    #                   'collision_count_dict': collision_count_dict,
    #                   'num_trials': num_trials}
    #
    # pickle_export(dirname_out, filename_out, stat_plot_data)

    # Pull in saved agg data
    path_in = os.path.join(dirname_out, filename_out)
    stat_plot_data = pickle_import(path_in)

    fig, ax = collision_stat_plot(stat_plot_data, collision_bound=0.10)
    filename_out = 'collision_stat_plot.pdf'
    path_out = os.path.join(dirname_out, filename_out)
    fig.savefig(path_out)

    # Zoom the x-axis to the region of interest
    ax.set_xlim([0.0004, 0.015])
    filename_out = 'collision_stat_plot_zoom.pdf'
    path_out = os.path.join(dirname_out, filename_out)
    fig.savefig(path_out)

    time_stats()
