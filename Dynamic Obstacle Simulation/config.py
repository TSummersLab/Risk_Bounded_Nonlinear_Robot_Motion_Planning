# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:59:37 2021

@author: vxr131730
Author: Venkatraman Renganathan
Email: vrengana@utdallas.edu
Github: https://github.com/venkatramanrenganathan

- Create a configuration file for RRT*. Functions that use RRT* outputs 
  will use some of these configurations

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
import numpy as np
import pickle

maxIter     = 100
Trial_Num   = 1
Trial_Total = 100
num_states  = 6
num_outputs = 4
total_dim   = num_states + num_outputs

# Flag to select the estimator
# 1: UKF, 0: EKF
estimatorSelector = 1

# Flag to select DR or CC Risk Estimation
# True - Use DR Risk Constraints
# False - Use Chance Constraints
DRFlag = True

# Tracking Horizon for Car
carTrackHorizon = 4 

# Flag to decide if dynamic obstacle or static obstacle based simulation
# True: Dynamic Obstacle, False: Static Obstacle
dynamicObs = True

# List of obstacle velocities
obsVelocities = [0.10, 0.20]

# Select velocitySelector to choose among the velocities list items
velocitySelector = 0 

# Based on the dynamicObs flag, decide the velocity of obstacle
if dynamicObs:
    constantObsVelocity = obsVelocities[velocitySelector]
else:
    constantObsVelocity = 0

