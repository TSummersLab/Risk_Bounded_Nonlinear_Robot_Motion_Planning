# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:16:25 2021

@author: vxr131730 - Venkatraman Renganathan

This code simulates the bicycle dynamics of car by steering it using NMPC -
nonlinear model predictive control (multiple shooting technique) and the state 
estimation using (UKF) unscented kalman filter. This code uses CARLA simulator.

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

import glob
import os
import sys
import math
import casadi as ca
import numpy as np
from numpy import random as npr
from numpy import linalg as LA
import State_Estimator as State_Estimator

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import config

###############################################################################
###############################################################################
#######################  Global Functions #####################################
###############################################################################
###############################################################################

class NMPC_Steering:
    
    def __init__(self, setup_params):
        """
        Constructor Function initializes the class with variable values populated
        using the setup_params

        Parameters
        ----------
        setup_params : Dictionary
            Dictionary containing the parameter values to run NMPC Steering.

        Returns
        -------
        None.

        """
    
        # Define Simulation Parameters 
        self.T           = setup_params["T"]           # Discretization Time Step
        self.N           = setup_params["N"]           # Prediction Horizon
        self.L           = setup_params["L"]           # Wheelbase of car
        self.sim_time    = setup_params["sim_time"]    # Total Simulation Time
        self.num_ctrls   = setup_params["num_ctrls"]   # Number of controls
        self.num_states  = setup_params["num_states"]  # Number of states
        self.num_outputs = setup_params["num_outputs"] # Number of outputs        
        self.goal_x      = setup_params["goal_x"]      # Goal x position
        self.goal_y      = setup_params["goal_y"]      # Goal y position        
        
        # CONSTRAINT BOUNDS
        min_accel = setup_params["min_accel"] # minimum throttle
        max_accel = setup_params["max_accel"] # maximum throttle
        min_steer = setup_params["min_steer"] # minimum steering angle 
        max_steer = setup_params["max_steer"] # maximum steering angle    
        min_x_pos = setup_params["min_x_pos"] # minimum position
        max_x_pos = setup_params["max_x_pos"] # maximum position
        min_y_pos = setup_params["min_y_pos"] # minimum position
        max_y_pos = setup_params["max_y_pos"] # maximum position
        min_veloc = setup_params["min_vel"]   # minimum velocity
        max_veloc = setup_params["max_vel"]   # maximum velocity
        min_x_obs = setup_params["obs_x_min"] # minimum obstacle x position
        max_x_obs = setup_params["obs_x_max"] # maximum obstacle x position
        min_y_obs = setup_params["obs_y_min"] # minimum obstacle y position
        max_y_obs = setup_params["obs_y_max"] # maximum obstacle y position
        
        
        # Define the State and Control Penalty matrices    
        self.Q = setup_params["Q"]
        self.R = setup_params["R"]
        
        # Define the noise means
        self.mu_w = setup_params["mu_w"] # Mean of process noises
        self.mu_v = setup_params["mu_v"] # Mean of sensor noises 
        
        # Define the joint covariance
        self.joint_mu  = setup_params["joint_mu"]
        self.joint_cov = setup_params["joint_cov"]
        
        # Define Covariance Matrices        
        self.SigmaE   = setup_params["SigmaE"]   # Estimation Error Covariance # 0.0001*np.identity(num_states)
        self.SigmaW   = setup_params["SigmaW"]   # Process Noise Covariance # np.diag([0.0005, 0.0005, 0, 0])
        self.SigmaV   = setup_params["SigmaV"]   # Sensor Noise Covariance  # 0.0001*np.identity(num_outputs)
        self.CrossCor = setup_params["CrossCor"] # Cross Correlation btw w_{t} annd v_{t}                 
        self.WV_Noise = setup_params["WV_Noise"]
        
        # Initiate an instance of opti class of casadi    
        self.opti = ca.Opti()
        
        # control variables, linear velocity v and angle velocity omega
        self.opt_controls = self.opti.variable(self.N, self.num_ctrls)
        self.opt_states   = self.opti.variable(self.N+1, self.num_states)
        accel        = self.opt_controls[:, 0]
        steer        = self.opt_controls[:, 1]    
        x            = self.opt_states[:, 0]
        y            = self.opt_states[:, 1]
        theta        = self.opt_states[:, 2]
        v            = self.opt_states[:, 3]
        ox           = self.opt_states[:, 4]
        oy           = self.opt_states[:, 5]
    
        # parameters
        self.opt_x0 = self.opti.parameter(self.num_states)
        self.opt_xs = self.opti.parameter(self.num_states)
        
        # create nonlinear model 
        self.f = lambda x_, u_: ca.vertcat(*[x_[3]*ca.cos(x_[2]), 
                                             x_[3]*ca.sin(x_[2]), 
                                             (x_[3]/self.L)*ca.tan(u_[1]),
                                             u_[0],
                                             0,
                                             0])
        self.f_np = lambda x_, u_, w_: np.array([x_[3]*np.cos(x_[2]) + w_[0], 
                                                 x_[3]*np.sin(x_[2]) + w_[1], 
                                                 (x_[3]/self.L)*np.tan(u_[1]) + w_[2],
                                                 u_[0] + w_[3],
                                                 0,
                                                 0])
        
        # Dynamics equality constraints - Denotes Multiple Shooting 
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)
        for i in range(self.N):
            x_next = self.opt_states[i,:] + self.T*self.f(self.opt_states[i,:], self.opt_controls[i,:]).T
            self.opti.subject_to(self.opt_states[i+1,:] == x_next)
    
        # Define the cost function
        self.obj = 0 
        for i in range(self.N):
            self.obj += ca.mtimes([(self.opt_states[i, :] - self.opt_xs.T), self.Q, (self.opt_states[i, :] - self.opt_xs.T).T]) + \
                   ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T])
        # TODO change cost func obj as sum of self.opt_states[i, :] - self.opt_xs[i, :].T  
    
        # Set the objective to be minimized
        self.opti.minimize(self.obj)
    
        # Define the state and control constraints
        self.opti.subject_to(self.opti.bounded(min_x_pos, x, max_x_pos))
        self.opti.subject_to(self.opti.bounded(min_y_pos, y, max_y_pos))
        self.opti.subject_to(self.opti.bounded(-ca.inf, theta, ca.inf))
        self.opti.subject_to(self.opti.bounded(min_veloc, v, max_veloc))
        self.opti.subject_to(self.opti.bounded(min_x_obs, ox, max_x_obs))
        self.opti.subject_to(self.opti.bounded(min_y_obs, oy, max_y_obs))
        self.opti.subject_to(self.opti.bounded(min_accel, accel, max_accel))
        self.opti.subject_to(self.opti.bounded(min_steer, steer, max_steer))
    
        # Define the solver settings
        opts_setting = {'ipopt.max_iter':2000, 
                        'ipopt.print_level':0, 
                        'print_time':0,   
                        'ipopt.acceptable_tol':1e-8, 
                        'ipopt.acceptable_obj_change_tol':1e-6}
    
        # Set the solver as IPOPT
        self.opti.solver('ipopt', opts_setting) 
    
    ###########################################################################
    ###########################################################################
    
    def MotionModel(self, x_, u_, w_): 
        """
        Returns a kinematic Bicycle motion model with additive disturbance.

        Parameters
        ----------
        x_ : numpy array 
            States.
        u_ : numpy array 
            controls.
        w_ : numpy array 
            disturbance.

        Returns
        -------
        motionModel : numpy array 
            kinematic bicycle model with additive distrubance.

        """
        # Infer the dimension
        dim = x_.shape[0]        
        
        # Prepare the motionModel        
        motionModel = np.zeros(dim)        
        motionModel[0] = x_[3]*np.cos(x_[2]) + w_[0]
        motionModel[1] = x_[3]*np.sin(x_[2]) + w_[1]
        motionModel[2] = (x_[3]/self.L)*np.tan(u_[1]) + w_[2]
        motionModel[3] = u_[0] + w_[3]
        
        return motionModel
    
    ###########################################################################
    ###########################################################################
    
    def MeasurementModel(self, newState): 
        """
        Returns a measurement model
        Totally 4 outputs are being returned by the measurement model
        [range distance to goal 
        bearing angle orientation with respect to goal
        obstacle x position * cos(6 deg)
        obstacle y position * cos(6 deg)]
        6 deg (roughly 0.1rad) is assumed to be a known rotation 

        Parameters
        ----------
        newState : numpy array
            true states.

        Returns
        -------
        output : list
            measurement model.

        """
        
        x_to_goal = newState[0] - self.goal_x
        y_to_goal = newState[1] - self.goal_y        
        output = [math.sqrt(x_to_goal**2 + y_to_goal**2), 
                  math.atan2(y_to_goal, x_to_goal) - newState[2],
                  newState[4]*np.cos(0.10),
                  newState[5]*np.cos(0.10)]
        
        return output
     
    ###########################################################################
    ###########################################################################

    def TransformTheta(self, theta):
        """
        Transforms a negative theta value to positive value
    
        Parameters
        ----------
        theta : Float 
            Yaw Orientation.
    
        Returns
        -------
        theta :Float
            Transformed Yaw Orientation.
    
        """
        if theta < 0:
            theta = 360 - abs(theta)
        return theta
    
    ###############################################################################
    ###############################################################################
    
    def Get_Carla_Steer_Input(self, steer_angle):
        """
        Given a steering angle in radians, returns the steering input between [-1,1]
        so that it can be applied to the car in the CARLA simulator.
        Max steering angle = 70 degrees = 1.22 radians
        Ref: https://github.com/carla-simulator/ros-bridge/blob/master/carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py
        
        Input:
            steer_angle: steering angle in radians
            
        Output:
            steer_input: steering input between [-1,1]
        """
        
        steer_input = (1/1.22)*steer_angle
        steer_input = np.fmax(np.fmin(steer_input, 1.0), -1.0)
        
        return steer_input
        
    
    ###############################################################################
    ###############################################################################
    
    def Get_Carla_Throttle_Input(self, accel):
        """
        Given an acceleration in m/s^2, returns the throttle input between [0,1]
        so that it can be applied to the car in the CARLA simulator.
        Max acceleration = 3.0 m/s^2
        Ref: https://github.com/carla-simulator/ros-bridge/blob/master/carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py
        
        Input:
            accel: steering angle in radians
            
        Output:
            throttle_input: steering input between [0,1]
        """
        if accel > 0:
            brake_input    = 0.0
            throttle_input = (1/3)*accel    
            throttle_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
        else:
            throttle_input = 0
            brake_input = (1/3)*accel    
            brake_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
            
        return throttle_input, brake_input
        
    
    ###############################################################################
    ###############################################################################
    
    def Prepare_CARLA_Controls(self, u_k):
        """
        Given a control input list, the function Prepare_CARLA_Controls prepares 
        control inputs to be appliedd on Car type vehicle in a Carla simulator
    
        Parameters
        ----------
        u_k : list of float values
            [accel, steer].
    
        Returns
        -------
        car_controls : list of float values
            [throttle, steer, brake].
    
        """
        
        throttle_input, brake_input = self.Get_Carla_Throttle_Input(u_k[0])
        steer_input = self.Get_Carla_Steer_Input(u_k[1])    
        car_controls = [throttle_input, steer_input, brake_input]
        
        return car_controls
    
    ###############################################################################
    ###############################################################################
        
    def NMPC_Track_Waypoint(self, ego_player, static_player, ref_iterator, start_w, next_w, waypts, SigmaE):
        """
        Given a starting and ending waypoints, NMPC_Track_Waypoint returns the 
        cost and steering result after applying NMPC based steering to the bicycle 
        model with state & input constraints. If no solution was found, a None 
        disctionary is returned. TO BE USED WHILE CAR IS TRACKING THE WAYPOINTS
        IN CARLA ENVIRONMENTS
    
        Parameters
        ----------
        ego_player : Carla vehicle
            Object representing the Ego car in the Carla environment.
        static_player : Carla vehicle
            Object representing the Obstacle car in the Carla environment..
        start_w : list
            Starting waypoint.
        next_w : list
            Next Waypoint.
        waypts : Numpy array 
            Next N waypoints 
        SigmaE : numpy array
            Starting Covariance.
    
        Returns
        -------
        steerOutput : Dictionary
            Dictionary of cost and steer status information.
    
        """
        
        
        #========== Start the NMPC Tracking Code ==============================
        # Start from the current startpoint
        init_state  = np.array([start_w[0], 
                                start_w[1], 
                                start_w[2], 
                                start_w[3], 
                                start_w[4], 
                                start_w[5]]) 
        
        # Update details of the next waypoint to be tracked using endpoint
        final_state = np.array([next_w[0], 
                                next_w[1], 
                                next_w[2], 
                                next_w[3], 
                                next_w[4], 
                                next_w[5]])   
            
        # Initialize the true_state same as the initial_state
        true_state = init_state.copy().reshape(-1,1)          
       
        # Set the final state as opt_xs on opti structure
        self.opti.set_value(self.opt_xs, final_state)     
        
        # Define the initial control inputs sequence for all N time steps ahead   
        u_sequence = np.zeros((self.N, self.num_ctrls))        
        
        # Create holders for sequence of next_states and current estimated state
        x_hat       = init_state.copy()  # Current estimated state            
        next_states = waypts.T # np.zeros((self.N+1, self.num_states)) # Sequence of next N+1 states - TODO TAKE IT FROM RRT*PLANNED TRAJS                
        
        # Set the UKF Parameter settings as a dictionary
        estimator_params = {"n_x": self.num_states,
                            "n_o": self.num_outputs,
                            "goal_x": self.goal_x,
                            "goal_y": self.goal_y,
                            "xTrue": true_state,                      
                            "SigmaW": self.SigmaW, 
                            "SigmaV": self.SigmaV,
                            "CrossCor": self.CrossCor,
                            "L"  : self.L, 
                            "dT": self.T} 
            
        
        
        # Set the parameter values - Update opt_x0 using x_hat
        self.opti.set_value(self.opt_x0, x_hat)
        
        # Set optimal control to the u_sequence
        self.opti.set_initial(self.opt_controls, u_sequence) # shape:(N, 2)              
        
        # Set the parameter opt_states to next_states
        self.opti.set_initial(self.opt_states, next_states)  # shape:(N+1, 3)                        
        
        
        # Solve the NMPC optimization problem            
        try:
            sol = self.opti.solve()
        except:
            # print('Steering NLP Failed')
            return [], [], []
        
        # Obtain the optimal control input sequence
        u_sequence = sol.value(self.opt_controls)
        
        
        
        # Infer the first control input in that sequence
        u_k = u_sequence[0,:]        
        
        # Get the predicted state trajectory for N time steps ahead
        next_states = sol.value(self.opt_states)                  
        
        # Generate the joint noise random variable           
        wv = self.WV_Noise[:, ref_iterator].reshape(-1,1)             
        # wv = npr.multivariate_normal(self.joint_mu, self.joint_cov).reshape(-1,1)         
        
        # Extract process noise and sensor noise from joint random variable
        w = wv[:self.num_states, :].reshape(-1,1) 
        v = wv[self.num_states:, :].reshape(-1,1)        

        # Prepare the carla control inputs
        car_controls = self.Prepare_CARLA_Controls(u_k)
        # print('throttle', round(car_controls[0],2), 
        #       'steer', round(car_controls[1],2),
        #       'brake', round(car_controls[2],2))
        
        # Apply the control input to the carla simulator -TODO NEED TO UNCOMMENT IT
        ego_player.apply_control(carla.VehicleControl(throttle = float(car_controls[0]), 
                                                      steer    = float(car_controls[1]),
                                                      brake    = float(car_controls[2])))
          
        # Obtain where car is in simulator as true_state - GROUND TRUTH  
        # GET IT FROM THE SENSOR DATA - INSTEAD OF DIRECTLY INFERRING FROM SIMULATOR            
        x_gt   = ego_player.get_transform().location.x
        y_gt   = ego_player.get_transform().location.y
        yaw_gt = self.TransformTheta(ego_player.get_transform().rotation.yaw)*np.pi/180
        v_gt   = LA.norm([ego_player.get_velocity().x, ego_player.get_velocity().y]) # TODO - GET SIGN OF VEL VECTOR BY COMPARING WITH PREV VELO VEC
        obs_x  = static_player.get_transform().location.x
        obs_y  = static_player.get_transform().location.y
        
        # Form the true state and add process noise
        true_state = np.array([x_gt, y_gt, yaw_gt, v_gt, obs_x, obs_y]).reshape(-1,1) + w
        
        # Create a measurement using the true state and the sensor noise
        y_k = np.array(self.MeasurementModel(true_state), dtype=object).reshape(-1,1) + v        
        
        # Update the estimator_params dictionary to call the UKF state estimator
        estimator_params["x_hat"]  = x_hat
        estimator_params["SigmaE"] = SigmaE  
        estimator_params["u_k"]    = u_k.T        
        estimator_params["y_k"]    = y_k
        
        # Call the Estimator Module to get the state estimate
        state_estimator  = State_Estimator.State_Estimator(estimator_params)
        estimator_output = state_estimator.Get_Estimate()
        
        
        
        # Unbox the UKF state estimate & covariance
        x_hat  = np.squeeze(estimator_output["x_hat"])
        SigmaE = estimator_output["SigmaE"]
        
        return u_k, x_hat, SigmaE
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################