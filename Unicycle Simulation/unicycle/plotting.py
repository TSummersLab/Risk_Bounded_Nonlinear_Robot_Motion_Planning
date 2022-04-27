#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
import matplotlib.animation as ani


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
    ax.set_title('Entrywise Gains')
    return fig, ax


def animate(t_hist, x_hist, u_hist, x_ref_hist=None, u_ref_hist=None, show_start=True, show_goal=True,
            title=None, repeat=False, fig_offset=None, axis_limits=[-20, 20, -10, 30],
            robot_w=1.2, robot_h=2.0, wheel_w=2.0, wheel_h=0.50):
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

    def update(i, pc_list_robot):
        pc_list_robot = draw_robot(x_hist[i], ax, pc_list_robot)
        return pc_list_robot

    anim = ani.FuncAnimation(fig, update, fargs=[pc_list_robot], frames=T-1, interval=1, blit=True, repeat=repeat)
    plt.show()
