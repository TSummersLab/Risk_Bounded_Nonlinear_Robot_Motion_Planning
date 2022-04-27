#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:07:10 2022

@author: venkat
"""

import matplotlib.pyplot as plt
import cvxpy
import math
import random
import pickle
import config
import numpy as np
import casadi as ca
from numpy import linalg as LA
from numpy import random as npr
from matplotlib.patches import Rectangle

# Vehicle parameters
LENGTH = 4.5*0.55  # [m]
WIDTH = 2.0*0.55  # [m]
BACKTOWHEEL = 1.0*0.55  # [m]
WHEEL_LEN = 0.3*0.55  # [m]
WHEEL_WIDTH = 0.2*0.55  # [m]
TREAD = 0.7*0.55  # [m]
WB = 2.5*0.55  # [m]

# Car parameters
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0) # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6        # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6       # minimum speed [m/s]
MAX_ACCEL = 1.0               # maximum accel [m/ss]

# Goal position - SIGN CHANGED FOR TRACKING
GOAL_POSITION = [-75.0, 75.0]

# Flags to decide to show plots and animation
show_plot      = True # Flag to decide if plot is to be shown
show_animation = True # Flag to decide if animation is to be shown


def plot_car(x, y, yaw, steer, vehicleFlag, cabcolor="-r", truckcolor="-k"):  

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, 
                          -WHEEL_WIDTH - TREAD, 
                          WHEEL_WIDTH - TREAD, 
                          WHEEL_WIDTH - TREAD, 
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y
    
    # Based on the vehicleFlag, set the color of the vehicle 
    if vehicleFlag:
        plt.fill(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-r", alpha=0.5, label="Ego-Vehicle Start Position")
    else:
        plt.fill(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-k", alpha=0.5, label="Obstacle Vehicle Start Position")
    
    # Plot the wheels
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

###############################################################################
###############################################################################

###############################################################################
###############################################################################

if __name__ == '__main__':
    
    # Infer the goal position
    goal = GOAL_POSITION   

    fig = plt.figure(figsize = [16,9])
    # create an axes object in the figure
    ax = fig.add_subplot(1, 1, 1)
    # Plot the reference paths for different velocities
    for k in range(len(config.obsVelocities)):
        
        # Import the pickle file containing the reference path data
        if config.DRFlag:
            if config.obsVelocities[k] == 0.10:
                filename = 'waypts_ref_data010.pkl'
            if config.obsVelocities[k] == 0.20:
                filename = 'waypts_ref_data020.pkl'
        else:        
            if config.obsVelocities[k] == 0.10:
                filename = 'cc_waypts_ref_data010.pkl'
            if config.obsVelocities[k] == 0.20:
                filename = 'cc_waypts_ref_data020.pkl'
        
        # Import the correct data
        infile = open(filename,'rb')
        PATH_POINTS = pickle.load(infile)
        PATH_POINTS[1,:] = -1*PATH_POINTS[1,:] 
        PATH_POINTS[2,:] = -1*PATH_POINTS[2,:] 
        PATH_POINTS[5,:] = -1*PATH_POINTS[5,:] 
        infile.close()
        
        # Extract x,y from the path_points
        cx = PATH_POINTS[0,:]
        cy = PATH_POINTS[1,:]
        
        # Prepare label string
        labelstring1 = "Reference trajectory: Obstacle Vel = "
        labelstring2 = str(config.obsVelocities[k])
        labelstring = labelstring1 + labelstring2
        ax.plot(cx, cy, "--", label=labelstring)
    
    # Plot the ego vehicle
    plot_car(PATH_POINTS[0,0], PATH_POINTS[1,0], PATH_POINTS[2,0], steer=0, vehicleFlag=True)
    
    # Plot the obstacle vehicle
    plot_car(PATH_POINTS[4,0], PATH_POINTS[5,0], PATH_POINTS[2,0], steer=0, vehicleFlag=False)
    
    # Plot goal region
    ax.add_artist(Rectangle((goal[0]-1, goal[1]-1), 2, 2 , lw=2, fc='b', 
                            ec='b', alpha = 0.3, label="Goal Region"))    
    # Plot start point
    ax.scatter(cx[0], cy[0], s=400, c='goldenrod', ec= 'k', linewidths=2, 
               marker='^', label='Start Point', zorder=20, alpha = 0.3)
    
    # Legend and labels and show plot
    ax.axis("equal")
    ax.axis('off')
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    ax.legend()    
    plt.show()