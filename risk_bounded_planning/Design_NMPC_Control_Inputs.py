#!/usr/bin/env python

###############################################################################
"""
@author: vxr131730 - Venkatraman Renganathan

This code simulates the bicycle dynamics of car by steering it using NMPC -
nonlinear model predictive control (multiple shooting technique) and the state 
estimation using (UKF) unscented kalman filter. 

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
import random
import math
import time
import casadi as ca
import numpy as np
import cv2
from casadi import *
from numpy import random as npr
from casadi.tools import *
from numpy import linalg as LA
import State_Estimator as State_Estimator


###############################################################################
###############################################################################
#######################  MAIN NMPC STEERING CODE ##############################
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
        self.dT          = setup_params["T"]           # Discretization Time Step
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
        self.QN = 10*self.Q
        
        # Define the noise means
        self.mu_w = setup_params["mu_w"] # Mean of process noises
        self.mu_v = setup_params["mu_v"] # Mean of sensor noises 
        
        # Define the joint covariance
        self.joint_mu  = setup_params["joint_mu"]
        self.joint_cov = setup_params["joint_cov"]
        
        # Define Covariance Matrices
        self.SigmaW   = setup_params["SigmaW"]   # Process Noise Covariance # np.diag([0.0005, 0.0005, 0, 0])
        self.SigmaV   = setup_params["SigmaV"]   # Sensor Noise Covariance  # 0.0001*np.identity(num_outputs)
        self.SigmaE   = setup_params["SigmaE"]   # Estimation Error Covariance # 0.0001*np.identity(num_states)
        self.CrossCor = setup_params["CrossCor"] # Cross Correlation btw w_{t} annd v_{t}                                 
        
        # Initiate an instance of opti class of casadi    
        self.opti = ca.Opti()
        
        # control variables, linear velocity v and angle velocity omega
        self.opt_controls = self.opti.variable(self.N, self.num_ctrls)
        self.opt_states = self.opti.variable(self.N+1, self.num_states)
        accel = self.opt_controls[:, 0]
        steer = self.opt_controls[:, 1]    
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]
        theta = self.opt_states[:, 2]
        v = self.opt_states[:, 3]
        ox = self.opt_states[:, 4]
        oy = self.opt_states[:, 5]     
    
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
                                                  0],dtype=object)        
        
        # Dynamics equality constraints - Denotes Multiple Shooting                                                           
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T)
        for i in range(self.N):            
            x_next = self.opt_states[i,:] + self.dT*self.f(self.opt_states[i,:], self.opt_controls[i,:]).T
            self.opti.subject_to(self.opt_states[i+1,:] == x_next)        
        # self.opti.subject_to(self.opt_states[self.N, :] == self.opt_xs.T) # Add final constraint
            
        # Define the cost function
        self.obj = 0 
        for i in range(self.N):
            self.obj += ca.mtimes([(self.opt_states[i, :] - self.opt_xs.T), self.Q, (self.opt_states[i, :] - self.opt_xs.T).T]) + \
                        ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T])            
        # self.obj += ca.mtimes([(self.opt_states[self.N, :] - self.opt_xs.T), self.QN, (self.opt_states[self.N, :] - self.opt_xs.T).T])   
    
        # Set the objective to be minimized
        self.opti.minimize(self.obj)
    
        # Define the state and control constraints
        self.opti.subject_to(self.opti.bounded(min_x_pos, x, max_x_pos))
        self.opti.subject_to(self.opti.bounded(min_y_pos, y, max_y_pos))
        self.opti.subject_to(self.opti.bounded(-ca.inf, theta, ca.inf)) # between 180 and 360
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
    
    def NMPC_Steer(self, start_w, next_w, SigmaE=None):
        """
        Given a starting and ending waypoints, NMPC_Steers returns the trajectory 
        obtained by applying NMPC based steering to the bicycle model with state
        and input constraints. If no solution was found, a None based disctionary
        is returned. TO BE USED WHILE BUILDING THE PATH PLANNING TREE

        Parameters
        ----------
        start_w : numpy array
            Starting waypoint.
        next_w : numpy array
            Ending Waypoint.
        SigmaE : numpy array, optional
            Starting Covariance. The default is None.

        Returns
        -------
        steerOutput : Dictionary
            Dictionary of trajectory values.

        """
        
        #========== Start the NMPC Tracking Code ==============================                                                                
        # Infer the Initial Covariance            
        SigmaE = start_w.covar[-1,:,:]
        
        # Start from the current startpoint
        init_state = np.squeeze(start_w.means[-1,:,:]) 
        
        # Update details of the next waypoint to be tracked using endpoint
        final_state = np.squeeze(next_w.means[-1,:,:])
            
        # Initialize the true_state same as the initial_state
        true_state = init_state.copy() 
        
        # Set the final state as opt_xs on opti structure
        self.opti.set_value(self.opt_xs, final_state)     
        
        # Define the initial control inputs sequence for all N time steps ahead   
        u_sequence = np.zeros((self.N, self.num_ctrls))
        
        # Create holders for sequence of next_states and current estimated state
        x_hat = init_state.copy()  # Current estimated state                  
        next_states = np.zeros((self.N+1, self.num_states)) # Sequence of next N+1 states
        
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
                            "dT": self.dT} 
        
        # Define place holders to store the history of control inputs and states
        u_k_hist = [] 
        x_k_hist = [x_hat] 
        S_k_hist = [SigmaE] # Need to pass this parameter
     
        # Start MPC Optimization
        mpc_iter = 0 # variable to count MPC iterations                        
        maxIters = 100  # maximum number of iterations
        cost_total = 0.0  # Total cost to go from source to destination
        terminal_cost = 0.0 # Current stage cost
        goal_toler = 0.5  # Desired distance tolerance to be closer to goal                
        w = np.zeros(self.num_states).reshape(-1,1) # # Tree Building Context, so noises = 0   
        v = np.zeros(self.num_outputs).reshape(-1,1) # # Tree Building Context, so noises = 0           
        
        # # OLD CODE - Revived
        # while True:                
        #     # Current waypoint tracking is epsilon-successful, track the next waypoint
        #     if LA.norm(next_states[-1, 0:2] - final_state[0:2]) <= goal_toler or mpc_iter > maxIters:                
        #         break        
            
        #     # Set the parameter values - Update opt_x0 using x_hat
        #     self.opti.set_value(self.opt_x0, x_hat)
            
        #     # Set optimal control to the u_sequence
        #     self.opti.set_initial(self.opt_controls, u_sequence) # shape:(N, 2)
            
        #     # Set the parameter opt_states to next_states
        #     self.opti.set_initial(self.opt_states, next_states)  # shape:(N+1, 3)
            
        #     # Solve the NMPC optimization problem
        #     try:                
        #         sol = self.opti.solve()                 
        #     except(RuntimeError):
        #         # raise exception('Infeasible point tracking !')                
        #         # print('Infeasible point tracking !')
        #         # Prepare output dictionary
        #         steerOutput = {"means": None, "covar": None, "cost" : 0, "status": False}
        #         return steerOutput
                
        #     # Increment the total cost by stage_cost = terminal_cost - sol.value(self.obj)
        #     if mpc_iter >= 1: 
        #         cost_total += terminal_cost - sol.value(self.obj)            
            
        #     # Update the terminal cost
        #     terminal_cost = sol.value(self.obj) # self.opti.debug.value(self.obj) 
            
        #     # Obtain the optimal control input sequence
        #     u_sequence = sol.value(self.opt_controls)
            
        #     # Infer the first control input in that sequence
        #     u_k = u_sequence[0,:]
            
        #     # Get the predicted state trajectory for N time steps ahead
        #     next_states = sol.value(self.opt_states)  
            
        #     # If the context is building the tree, then use kinematic model                                 
        #     true_state = true_state + self.dT*self.f_np(true_state, u_k, w)
            
        #     # Create a measurement using the true state and the sensor noise
        #     y_k = np.array(self.MeasurementModel(true_state), dtype=object).reshape(-1,1) + v
            
        #     # Update the estimator_params dictionary to call the UKF state estimator
        #     estimator_params["x_hat"]  = x_hat
        #     estimator_params["u_k"]    = u_k.T
        #     estimator_params["SigmaE"] = SigmaE  
        #     estimator_params["y_k"]    = y_k
            
        #     # Call the Estimator Module to get the state estimate
        #     state_estimator  = State_Estimator.State_Estimator(estimator_params)
        #     estimator_output = state_estimator.Get_Estimate()
            
        #     # Unbox the UKF state estimate & covariance
        #     x_hat  = np.squeeze(estimator_output["x_hat"])
        #     SigmaE = estimator_output["SigmaE"]
            
        #     # Prepare u_sequence and next_states for next loop initialization
        #     # Since 1st entry is already used, trim it and repeat last entry to fill the vector
        #     # Eg. if u_sequence = [1 2 3 4 5 6], then fill as u_sequence = [2 3 4 5 6 6]
        #     # Why? u_sequence, next_states: best guesses for opt_controls, opt_states resp. in next iteration
        #     lastBestUSeq = u_sequence
        #     lastBestNextStates = next_states
        #     u_sequence  = np.concatenate((u_sequence[1:], u_sequence[-1:]))
        #     next_states = np.concatenate((next_states[1:], next_states[-1:]))
            
        #     # Store the history
        #     u_k_hist.append(u_k)
        #     x_k_hist.append(next_states[0,:])
        #     S_k_hist.append(SigmaE) 
            
        #     # Increment the mpc loop counter
        #     mpc_iter = mpc_iter + 1 
            
        # # Since visitingGoalDistance < goalTolerance, NLP has solved successfully
        # # Simulate for N timesteps ahead and just collect the history
        # for k in range(0, self.N):                   
        
        #     # Infer the first control input in that sequence
        #     u_k = lastBestUSeq[k,:]
            
        #     # If the context is building the tree, then use kinematic model                                 
        #     true_state = true_state + self.dT*self.f_np(lastBestNextStates[k,:], u_k, w)
            
        #     # Create a measurement using the true state and the sensor noise
        #     y_k = np.array(self.MeasurementModel(true_state), dtype=object).reshape(-1,1) + v                           
            
        #     # Update the estimator_params dictionary to call the UKF state estimator
        #     estimator_params["x_hat"]  = x_hat
        #     estimator_params["u_k"]    = u_k.T
        #     estimator_params["SigmaE"] = SigmaE  
        #     estimator_params["y_k"]    = y_k
            
        #     # Call the Estimator Module to get the state estimate
        #     state_estimator  = State_Estimator.State_Estimator(estimator_params)
        #     estimator_output = state_estimator.Get_Estimate()
            
        #     # Unbox the UKF state estimate & covariance
        #     x_hat  = np.squeeze(estimator_output["x_hat"])
        #     SigmaE = estimator_output["SigmaE"]                
            
        #     # Store the history                
        #     x_k_hist.append(x_hat)
        #     S_k_hist.append(SigmaE) 
        
        # # Since things were successfull, append the final state as if it reached perfectly
        # # x_k_hist.append(final_state)
        # # S_k_hist.append(SigmaE) # Same covar as the last covar
            
        # # Prepare output dictionary to return
        # steerOutput = {"means": x_k_hist,
        #                 "covar": S_k_hist,
        #                 "cost" : cost_total,
        #                 "status": True}
        
        # return steerOutput 
        
        
        # NEW CODE
        # Set the parameter values - Update opt_x0 using x_hat
        self.opti.set_value(self.opt_x0, x_hat)
        
        # Set optimal control to the u_sequence
        self.opti.set_initial(self.opt_controls, u_sequence) # shape:(N, 2)
        
        # Set the parameter opt_states to next_states
        self.opti.set_initial(self.opt_states, next_states)  # shape:(N+1, 3)
        
        # Solve the NMPC optimization problem
        try:                
            sol = self.opti.solve()                 
        except(RuntimeError):
            # raise exception('Infeasible NLP Problem !')                
            # print('Infeasible NLP Problem !')            
            # Prepare output dictionary
            steerOutput = {"means": None, "covar": None, "cost" : 0, "status": False}
            return steerOutput
        
        # Update the cost_total
        cost_total = sol.value(self.obj) # self.opti.debug.value(self.obj) 
        
        # Obtain the optimal control input sequence
        u_sequence = sol.value(self.opt_controls)        
        
        # Get the predicted state trajectory for N time steps ahead
        next_states = sol.value(self.opt_states)         
        
        # Get the last simulated state
        simulatedLastState = next_states[-1,:]
        
        for tim in range(next_states.shape[0]):            
            if int(next_states[tim,0]) == 0 or int(next_states[tim,1]) == 0:
                steerOutput = {"means": None, "covar": None, "cost" : 0, "status": False}
                return steerOutput
        
        # print('next_states', np.round_(next_states,2))
        # print('simulatedLastState', np.round_(simulatedLastState[0:2],2))        
        
        # NMPC Succeeded - Simulate for N timesteps ahead
        for k in range(0, self.N):                   
        
            # Infer the first control input in that sequence
            u_k = u_sequence[k,:]
            
            # If the context is building the tree, then use kinematic model                                 
            true_state = true_state + self.dT*self.MotionModel(next_states[k,:], u_k, w)            
            
            # Create a measurement using the true state and the sensor noise
            y_k = np.array(self.MeasurementModel(true_state)).reshape(-1,1) + v                
            
            # Update the estimator_params dictionary to call the UKF state estimator
            estimator_params["x_hat"] = next_states[k,:]
            estimator_params["u_k"] = u_k.T
            estimator_params["SigmaE"] = SigmaE  
            estimator_params["y_k"] = y_k
            
            # Call the Estimator Module to get the state estimate
            state_estimator  = State_Estimator.State_Estimator(estimator_params)
            estimator_output = state_estimator.Get_Estimate()
            
            # Unbox the UKF state estimate & covariance            
            SigmaE = estimator_output["SigmaE"]                
            
            # Store the history                            
            x_k_hist.append(next_states[k,:])
            S_k_hist.append(SigmaE) 
        
        # Prepare output dictionary to return
        steerOutput = {"means": x_k_hist, "covar": S_k_hist,
                        "cost" : cost_total, "status": True}
        
        return steerOutput            
    
        
        
    
        # # OLD CODE
        # while True:    

        #     print('NMPC Iteration:', mpc_iter, 'with distance to goal =', round(dist_to_goal, 2))              
            
        #     # Current waypoint tracking is successful, track the next waypoint
        #     if dist_to_goal <= goal_toler or mpc_iter > maxIters:
        #         break        
            
        #     # Set the parameter values - Update opt_x0 using x_hat
        #     self.opti.set_value(self.opt_x0, x_hat)
            
        #     # Set optimal control to the u_sequence
        #     self.opti.set_initial(self.opt_controls, u_sequence) # shape:(N, 2)
            
        #     # Set the parameter opt_states to next_states
        #     self.opti.set_initial(self.opt_states, next_states)  # shape:(N+1, 3)
            
        #     # Solve the NMPC optimization problem
        #     try:                
        #         sol = self.opti.solve()                 
        #     except(RuntimeError):
        #         # raise exception('Infeasible point tracking !')                
        #         print('Infeasible point tracking !')
        #         # Prepare output dictionary
        #         steerOutput = {"means": None, "covar": None, "cost" : 0, "status": False}
        #         return steerOutput
                
        #     # Increment the total cost by stage_cost = terminal_cost - sol.value(self.obj)
        #     if mpc_iter >= 1: 
        #         cost_total += terminal_cost - sol.value(self.obj)
            
        #     # Update the terminal cost
        #     terminal_cost = sol.value(self.obj) # self.opti.debug.value(self.obj) 
            
        #     # Obtain the optimal control input sequence
        #     u_sequence = sol.value(self.opt_controls)
            
        #     # Infer the first control input in that sequence
        #     u_k = u_sequence[0,:]
            
        #     # Get the predicted state trajectory for N time steps ahead
        #     next_states = sol.value(self.opt_states) 
        #     print(next_states)

        #     # Tree Building Context, so noises = 0                
        #     w = np.zeros(self.num_states).reshape(-1,1)
        #     v = np.zeros(self.num_outputs).reshape(-1,1)                    
            
        #     # If the context is building the tree, then use kinematic model                                 
        #     true_state = true_state + self.dT*self.MotionModel(true_state, u_k, w)
            
        #     # Create a measurement using the true state and the sensor noise
        #     y_k = np.array(self.MeasurementModel(true_state)).reshape(-1,1) + v
            
        #     # Update the estimator_params dictionary to call the UKF state estimator
        #     estimator_params["x_hat"]  = x_hat
        #     estimator_params["u_k"]    = u_k.T
        #     estimator_params["SigmaE"] = SigmaE  
        #     estimator_params["y_k"]    = y_k
            
        #     # Call the Estimator Module to get the state estimate
        #     state_estimator  = State_Estimator.State_Estimator(estimator_params)
        #     estimator_output = state_estimator.Get_Estimate()
            
        #     # Unbox the UKF state estimate & covariance
        #     x_hat  = np.squeeze(estimator_output["x_hat"])
        #     SigmaE = estimator_output["SigmaE"]
            
        #     # Prepare u_sequence and next_states for next loop initialization
        #     # Since 1st entry is already used, trim it and repeat last entry to fill the vector
        #     # Eg. if u_sequence = [1 2 3 4 5 6], then fill as u_sequence = [2 3 4 5 6 6]
        #     # Why? u_sequence, next_states: best guesses for opt_controls, opt_states resp. in next iteration
        #     u_sequence  = np.concatenate((u_sequence[1:], u_sequence[-1:]))
        #     next_states = np.concatenate((next_states[1:], next_states[-1:]))
            
        #     # Store the history
        #     u_k_hist.append(u_k)
        #     x_k_hist.append(x_hat)
        #     S_k_hist.append(SigmaE) 
            
        #     # Update distance to goal
        #     dist_to_goal = np.linalg.norm(x_hat[0:3] - final_state[0:3])
            
        #     # Increment the mpc loop counter
        #     mpc_iter = mpc_iter + 1             
            
    # ###################### Simulation Loop END ############################
    # # Print the trajectory information
    # # fmt = '{:<8}{:<10}{}' 
    # # print(fmt.format('Traj Node', 'x', 'y'))
    # # for k in range(len(x_k_hist)):
    # #     k_node = x_k_hist[k]
    # #     print(fmt.format(k, round(k_node[0], 2), round(k_node[1], 2)))                       
        

    ###########################################################################
    ###########################################################################
    
    