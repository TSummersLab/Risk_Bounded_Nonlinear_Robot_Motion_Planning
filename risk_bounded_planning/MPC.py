#!/usr/bin/env python3

"""
@author: vxr131730 - Venkatraman Renganathan

This code defines the class for using NMPC - nonlinear model predictive control 

CARLA SIMULATOR VERSION - 0.9.10
PYTHON VERSION          - 3.6.8
VISUAL STUDIO VERSION   - 2017
UNREAL ENGINE VERSION   - 4.24.3

This script is tested in Python 3.7.6, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2021.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3.7, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""

###############################################################################
####################### Import all the required libraries #####################
###############################################################################

import numpy as np
import NMPC_Params as NMPC_Params
import Design_NMPC_Control_Inputs as NMPC_Controller

###############################################################################
###############################################################################

class MPC:
    def __init__(self, sim_params):
        """
        Constructor function initilizing class variables using sim_params

        Parameters
        ----------
        sim_params : DICTIONARY
            Dictionary containing steering law parameters.

        Returns
        -------
        None.

        """

        # Instantiate the MPCParams class object to access its variables
        self.nmpc_params = NMPC_Params.NMPCParams(sim_params)      
                
        # States
        self.x   = sim_params["x"]
        self.y   = sim_params["y"]
        self.yaw = sim_params["yaw"]
        self.v   = sim_params["v"]
        self.ox  = sim_params["obstacle"][0]
        self.oy  = sim_params["obstacle"][1]

        # Wheel base        
        self.L  = sim_params["L"]
        
        # Acceleration bounds
        self.a_max = self.nmpc_params.a_max
        self.a_min = self.nmpc_params.a_min
        
        # Prepare the setup_params for the NMPC controller
        setup_params = {}
        setup_params["N"] = self.nmpc_params.len_horizon
        setup_params["T"] = self.nmpc_params.T
        setup_params["L"] = self.nmpc_params.L
        setup_params["Q"] = self.nmpc_params.Q
        setup_params["R"] = self.nmpc_params.R
        setup_params["mu_w"] = self.nmpc_params.mu_w
        setup_params["mu_v"] = self.nmpc_params.mu_v        
        setup_params["SigmaW"] = self.nmpc_params.SigmaW
        setup_params["SigmaV"] = self.nmpc_params.SigmaV
        setup_params["SigmaE"] = self.nmpc_params.SigmaE 
        setup_params["CrossCor"] = self.nmpc_params.CrossCorel 
        setup_params["joint_mu"] = self.nmpc_params.joint_mu
        setup_params["joint_cov"] = self.nmpc_params.joint_cov
        setup_params["min_vel"] = self.nmpc_params.v_min
        setup_params["max_vel"] = self.nmpc_params.v_max
        setup_params["min_accel"] = self.nmpc_params.a_min
        setup_params["max_accel"] = self.nmpc_params.a_max
        setup_params["min_steer"] = self.nmpc_params.min_steer
        setup_params["max_steer"] = self.nmpc_params.max_steer
        setup_params["min_x_pos"] = self.nmpc_params.x_min
        setup_params["min_y_pos"] = self.nmpc_params.y_min
        setup_params["max_x_pos"] = self.nmpc_params.x_max
        setup_params["max_y_pos"] = self.nmpc_params.y_max
        setup_params["obs_x_min"] = self.nmpc_params.obs_x_min
        setup_params["obs_y_min"] = self.nmpc_params.obs_y_min
        setup_params["obs_x_max"] = self.nmpc_params.obs_x_max
        setup_params["obs_y_max"] = self.nmpc_params.obs_y_max
        setup_params["goal_x"] = sim_params["goal_x"]
        setup_params["goal_y"] = sim_params["goal_y"]        
        setup_params["sim_time"] = self.nmpc_params.sim_time
        setup_params["num_ctrls"] = self.nmpc_params.num_ctrls
        setup_params["num_states"] = self.nmpc_params.num_states
        setup_params["num_outputs"] = self.nmpc_params.num_outputs         
        
        # Instantiate the NMPC controller for tree building
        self.nmpc_controller = NMPC_Controller.NMPC_Steering(setup_params)        
    
###############################################################################
###############################################################################
        
class Controller(object):
    def __init__(self, controller_params):
        """
        Constructor function initilizing class variables using controller_params

        Parameters
        ----------
        controller_params : DICTIONARY
            Dictionary containing steering law parameters..

        Returns
        -------
        None.

        """
        
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._obstacle_x         = 0
        self._obstacle_y         = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0        
        self._conv_rad_to_steer  = ((180.0/70.0)/np.pi)  # = 0.8185 = 1/1.222, where 1.22rad = 70deg = max_steer
                
        
        # Plug in a NMPC Controller - Can also plugin a different controller
        # Prepare simulation to be passed in for MPC steering 
        sim_params                 = {}
        sim_params["x"]            = self._current_x
        sim_params["y"]            = self._current_y
        sim_params["yaw"]          = self._current_yaw
        sim_params["v"]            = self._current_speed
        sim_params["obstacle"]     = controller_params["obstacle"]        
        sim_params["goal_x"]       = controller_params["goal_x_pos"]
        sim_params["goal_y"]       = controller_params["goal_y_pos"]
        sim_params["L"]            = controller_params["wheelbase"]                        
        sim_params["boundary"]     = controller_params["boundary"] # bottom_left [x,y, width, height]
        
        # Pass the dictionary parameters to initialize a MPC controller object
        # For tree building, instantiate a MPC class object
        self.controller = MPC(sim_params) 
###############################################################################
###############################################################################