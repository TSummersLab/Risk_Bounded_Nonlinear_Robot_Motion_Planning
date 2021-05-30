# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:41:30 2021

@author: vxr131730
"""
###############################################################################
####################### Import all the required libraries #####################
###############################################################################

import numpy as np
import MPC as MPC
from Path_Plotter import Plotter
import DR_RRTStar_Planner as DR_RRTStar_Planner
import Pickle_File_Preparer as Pickler

###############################################################################
####################### Define Global Functions ###############################
###############################################################################

def Wrap_Angle(angle_in_degree):
    """
    This function converts the angle input values in degree to radian values
    between [-pi, pi] and returns the resulting radian angle value

    Parameters
    ----------
    angle_in_degree : FLOAT
        Angle value in degrees.

    Returns
    -------
    angle_in_rad : FLOAT
        Angle value in radians.

    """
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while angle_in_rad > np.pi:
        angle_in_rad -= 2*np.pi
    while angle_in_rad <= -np.pi:
        angle_in_rad += 2*np.pi
    return angle_in_rad

###############################################################################
####################### Define Planner Parameters #############################
# THESE PARAMETERS REPRESENT THE CARLA ENVIRONMENT ENTITY PARAMETER VALUES.
# PARAMETER VALUES CUSTOMIZED TO OUR MOTION PLANNING PROBLEM SETTING.
###############################################################################


# Ego Vehicle
xEgoCar = -75.0  # Ego Vehicle X Position
yEgoCar = -45.0  # Ego Vehicle Y Position
zEgoCar = 0.0    # Ego Vehicle Z Position
rollEgo  = 0.0   # Ego Vehicle Roll Orientation 
pitchEgo = 0.0   # Ego Vehicle Pitch Orientation 
yawEgo   = 270   # Ego Vehicle Yaw Orientation 
lenEgoCar = 2.43 # Ego Vehicle Length
widEgoCar = 1.02 # Ego Vehicle Width


# Static Police Vehicle
xStatic  = -75.0   # Static Vehicle X Position
yStatic  = -65.0   # Static Vehicle Y Position
zStatic  = 0.0     # Static Vehicle Z Position
rollStatic  = 0.0  # Static Vehicle Roll Orientation 
pitchStatic = 0.0  # Static Vehicle Pitch Orientation 
yawStatic   = 270  # Static Vehicle Yaw Orientation 
lenStatic   = 2.49 # Static Vehicle Length
widStatic   = 1.02 # Static Vehicle Width

# Goal Position
xGoal   = -75.0  # Goal X Position
yGoal   = -75.0  # Goal Y Position
zGoal   = 1.0    # Goal Z Position
yawGoal = yawEgo # Goal Yaw Orientation 

# Boundary Information
bnd_x = -81.0  # Boundary Bottom Left X position
bnd_y = yEgoCar-lenEgoCar  # Boundary Bottom Left Y position = -45-2.43=-47.43
bnd_w = 7.0   # Boundary Width
bnd_h = -35.0  # Boundary Height #  -75

# =====================================================================
# =========== COMPUTE A MOTION PLAN FROM SOURCE TO DESTINATION ========
# =====================================================================

# Start from hero car position 
start_w = [xEgoCar, yEgoCar, yawEgo, 0] 
# End at goal position with same yawOrientation as starting point
end_w = [xGoal, yGoal, yawEgo, 0]
# Infer the position of obstacle
obs_w = [xStatic, yStatic, Wrap_Angle(yawStatic), widStatic, lenStatic]
# Infer the boundary information
boundaryInfo = [bnd_x, bnd_y, bnd_w, bnd_h]
# Infer environmental information
environmentInfo = [-83, -42.5, 10, -42.5]

# Set the controller Parameter settings as a dictionary
controller_params               = {}
controller_params["treeFlag"]   = True
controller_params["wheelbase"]  = 2.9
controller_params["obstacle"]   = obs_w  
controller_params["goal_x_pos"] = xGoal
controller_params["goal_y_pos"] = yGoal
controller_params["boundary"]   = boundaryInfo  # bottom_left [x,y, width, height]

# Set the controller as MPC Controller with controller_params
# Get an object of the controller and store it as self.controller
controllerObj = MPC.Controller(controller_params)

# Set the planner Parameter settings as a dictionary
planner_params             = {}
planner_params["start_w"]  = start_w
planner_params["end_w"]    = end_w
planner_params["obs_w"]    = obs_w 
planner_params["steerLaw"] = controllerObj
planner_params["maxIter"]  = 100
planner_params["desDist"]  = 5.0
planner_params["EgoDim"]   = [2.43, 1.02] # length = 2.43, width = 1.02 got from carla
planner_params["boundary"] = boundaryInfo # format: bottom_left [x,y, width, height]
planner_params["envInfo"]  = environmentInfo # format: bottom_left [x,y, width, height]

# Call the motion planner to plan the path from source to goal
DR_RRTStar_Planner.Planner(planner_params)

# Pickle all the files
Pickler.Save_Data(xGoal, yGoal)

# Plot the solution reference trajectory and the tree
boundaryInfo = [-83, -42.5, 10, -42.5]
Plotter(start_w, end_w, obs_w, boundaryInfo) 
