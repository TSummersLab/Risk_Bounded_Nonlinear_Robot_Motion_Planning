#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:34:23 2022

@author: venkat
"""

"""
Path tracking simulation with nonlinear model predictive control & UKF
"""
import matplotlib.pyplot as plt
import math
import random
import pickle
import config
import numpy as np
import casadi as ca
from numpy import linalg as LA
from numpy import random as npr
from matplotlib.patches import Rectangle

###############################################################################
###############################################################################

DT = 0.1  # [s] time tick
constantObsVelocity = config.constantObsVelocity

# Vehicle parameters
LENGTH = 4.5*0.55  # [m] agrees with carla vehicle length 2.49m
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
MAX_TIME = 300.0              # max simulation time

# Goal position - SIGN CHANGED FOR TRACKING
GOAL_POSITION = [-75.0, 75.0]

# Flags to decide to show plots and animation
show_plot      = True # Flag to decide if plot is to be shown
show_animation = True # Flag to decide if animation is to be shown

# Import the pickle file containing the reference path data
if config.DRFlag:
    if config.obsVelocities[config.velocitySelector] == 0.10:
        filename = 'waypts_ref_data010.pkl'
    if config.obsVelocities[config.velocitySelector] == 0.20:
        filename = 'waypts_ref_data020.pkl'
else:        
    filename = 'cc_waypts_ref_data.pkl'
infile = open(filename,'rb')
PATH_POINTS = pickle.load(infile)
PATH_POINTS[1,:] = -1*PATH_POINTS[1,:] 
PATH_POINTS[2,:] = -1*PATH_POINTS[2,:] 
PATH_POINTS[5,:] = -1*PATH_POINTS[5,:] 
infile.close()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class State_Estimator:
    
    def __init__(self, estimator_params):
        
        self.params    = estimator_params
        self.estimates = None
        
        # Plug in an Estimator based on config selection
        self.estimator = UKF() 
        

    ###########################################################################

    def Get_Estimate(self):
        
        estimates = self.estimator.Estimate(self.params)
        
        return estimates

###############################################################################
###############################################################################
################## UNSCENTED KALMAN FILTER IMPLEMENTATION #####################
###############################################################################
###############################################################################

class UKF:
    
    def __init__(self):
        
        # Instantiate the class variables
        self.zMean         = 0 
        self.u_k           = 0     
        self.zCovar        = 0
        self.n_x           = 0
        self.n_o           = 0
        self.SigmaW        = 0
        self.SigmaV        = 0
        self.CrossCor      = 0
        self.y_k           = 0
        self.L             = 0
        self.dT            = 0
        self.Wc            = 0
        self.Wm            = 0
        self.alpha         = 0
        self.beta          = 0
        self.n             = 0
        self.kappa         = 0
        self.lambda_       = 0
        self.num_sigma_pts = 0
        self.Tt            = 0
        self.knownIP       = 0
        self.goal_x        = 0
        self.goal_y        = 0
        
    ###########################################################################
        
    def Get_Weight_Matrices(self):
        
        # Initialize Van der Merwe's weighting matrix
        self.Wc = np.zeros((self.num_sigma_pts, 1))
        self.Wm = np.zeros((self.num_sigma_pts, 1))    
        
        # Compute the Van der Merwe's weighting matrix values    
        for i in range(self.num_sigma_pts):
            if i == 0:
                self.Wc[i,:] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
                self.Wm[i,:] = self.lambda_ / (self.n + self.lambda_)
                continue
            self.Wc[i,:] = 1/(2*(self.n + self.lambda_))
            self.Wm[i,:] = 1/(2*(self.n + self.lambda_))            
            
    ###########################################################################
            
    def Perform_Decorrelation(self):        
        
        # Perform Decorrelation Scheme
        self.Tt = self.CrossCor @ LA.inv(self.SigmaV)    
        
        # Covariance of new process noise which has 0 cross-correlation with sensor noise
        self.SigmaW = self.SigmaW - self.Tt @ self.SigmaV @ self.Tt.T        
                
        # Fabricate the known input = Tt*y_hat
        self.knownIP = np.squeeze(self.Tt @ self.y_k)
        # self.knownIP = np.squeeze(self.Tt @ self.MeasurementModel(self.zMean).reshape(-1,1))
        
            
    ###########################################################################
    
    def Generate_Sigma_Points(self):
        
        # Define the direction matrix        
        U = LA.cholesky((self.n + self.lambda_)*self.zCovar)     
        
        # Generate the sigma points using Van der Merwe algorithm
        # Define Place holder for all sigma points
        sigmaPoints = np.zeros((self.n, self.num_sigma_pts))    
        
        # First SigmaPoint is always the mean 
        sigmaPoints[:,0] = self.zMean.T 
        
        # Generate sigmapoints symmetrically around the mean
        for k in range(self.n): 
            sigmaPoints[:, k+1]        = sigmaPoints[:,0] + U[:, k]
            sigmaPoints[:, self.n+k+1] = sigmaPoints[:,0] - U[:, k]
            
        return sigmaPoints
        
    ###########################################################################    
    
    def PredictionStep(self, sigmaPoints):        
            
        # Get the shape of sigmaPoints
        ro, co = np.shape(sigmaPoints)
        
        # Create the data structure to hold the transformed points
        aprioriPoints = np.zeros((ro, co))
        
        # Loop through and pass each and every sigmapoint
        for i in range(co):
            aprioriPoints[:, i] = self.NewMotionModel(sigmaPoints[:, i])    
        
        # Compute the mean and covariance of the transformed points
        aprioriOutput = self.ComputeStatistics(aprioriPoints, apriori_Flag = 1)
        
        # Add the aprioriPoints to output
        aprioriOutput["aprioriPoints"] = aprioriPoints 
        
        return aprioriOutput
    
    ###########################################################################
    
    def UpdateStep(self, aprioriPoints):        
        
        # Get the shape of aprioriPoints
        ro, M = np.shape(aprioriPoints)
        
        # Get the number of outputs
        num_outputs = self.n_o
           
        # Create the data structure to hold the transformed points
        aposterioriPoints = np.zeros((num_outputs, M)) # 4 states, 2 outputs
        
        # Loop through and pass each and every sigmapoint
        for i in range(M):
            aposterioriPoints[:, i] = self.MeasurementModel(aprioriPoints[:, i])
        
        # Compute the mean and covariance of the transformed points    
        aposterioriOutput = self.ComputeStatistics(aposterioriPoints, apriori_Flag = 0)
        
        # Add the aposterioriPoints to the output dictionary
        aposterioriOutput["aposterioriPoints"] = aposterioriPoints
        
        return aposterioriOutput
    
    ###########################################################################
    def NewMotionModel(self, oldState):
        
        # newState = self.MotionModel(oldState) - self.Tt @ self.MeasurementModel(oldState) + self.knownIP
        newState = self.MotionModel(oldState)
        
        return newState
        
    
    ###########################################################################
    def MotionModel(self, oldState):   
         
        newState = oldState + [self.dT*oldState[3]*np.cos(oldState[2]), 
                               self.dT*oldState[3]*np.sin(oldState[2]), 
                               self.dT*(oldState[3]/self.L)*np.tan(self.u_k[1]),
                               self.dT*self.u_k[0],
                               0,
                               self.dT*constantObsVelocity]
        
        return newState
    
    ###########################################################################
    
    def MeasurementModel(self, newState):  
        """
        Totally 4 outputs are being returned by the measurement model
        [range distance to goal 
        bearing angle orientation with respect to goal
        obstacle x position * cos(6 deg)
        obstacle y position * cos(6 deg)]
        """
        
        x_to_goal = newState[0] - self.goal_x
        y_to_goal = newState[1] - self.goal_y        
        output = [math.sqrt(x_to_goal**2 + y_to_goal**2), 
                  math.atan2(y_to_goal, x_to_goal) - newState[2],
                  newState[4]*np.cos(0.10),
                  newState[5]*np.sin(0.10)] # TODO BIG CHANGE DONE - CHANGES COS TO SIN
        
        return output
    
    ###########################################################################
    
    def ComputeCrossCovariance(self, funParam):        
        
        # Compute the crossCovarMatrix    
        input1Shape = np.shape(funParam["input1"]) 
        input2Shape = np.shape(funParam["input2"])
        P = np.zeros((input1Shape[0], input2Shape[0]))
        
        for k in range(input1Shape[1]):        
            diff1 = funParam["input1"][:,k] - funParam["input1Mean"]
            diff2 = funParam["input2"][:,k] - funParam["input2Mean"]       
            P += funParam["weightMatrix"][k]*np.outer(diff1, diff2)
        
        return P
    
    ###########################################################################
    
    def ComputeStatistics(self, inputPoints, apriori_Flag):
        
        # Compute the weighted mean   
        inputPointsMean = np.dot(self.Wm[:,0], inputPoints.T)
        
        # Compute the weighted covariance
        inputShape = np.shape(inputPoints)
        P = np.zeros((inputShape[0], inputShape[0]))
        
        # Find the weighted covariance
        for k in range(inputShape[1]):        
            y = inputPoints[:, k] - inputPointsMean        
            P = P + self.Wc[k] * np.outer(y, y) 
        
        # Add the noise covariance
        if apriori_Flag == 1:            
            P += self.SigmaW
        if apriori_Flag == 0:            
            P += self.SigmaV    
        
        # Box the Output data
        statsOutput = {"mean": inputPointsMean, "Covar": P}
        
        return statsOutput
    
    ###########################################################################
    
    def Compute_UKF_Gain(self, funParam, aposterioriCovar):
        
        # Compute the cross covariance matrix 
        crossCovarMatrix = self.ComputeCrossCovariance(funParam)
        
        # Compute Unscented Kalman Gain
        uKFGain = np.dot(crossCovarMatrix, LA.inv(aposterioriCovar))
        
        return uKFGain
        
    ###########################################################################    
     
    def Estimate(self, ukf_params):
        
        # Unbox the input parameters
        self.zMean    = ukf_params["x_hat"] 
        self.u_k      = ukf_params["u_k"]     
        self.zCovar   = ukf_params["SigmaE"]
        self.n_x      = ukf_params["n_x"]
        self.n_o      = ukf_params["n_o"]
        self.SigmaW   = ukf_params["SigmaW"] 
        self.SigmaV   = ukf_params["SigmaV"] 
        self.CrossCor = ukf_params["CrossCor"]
        self.y_k      = ukf_params["y_k"] 
        self.L        = ukf_params["L"] 
        self.dT       = ukf_params["dT"] 
        self.goal_x   = ukf_params["goal_x"]
        self.goal_y   = ukf_params["goal_y"]
        
        # Set the global variables
        self.alpha         = 1.0
        self.beta          = 2.0
        self.n             = self.n_x
        self.kappa         = 3 - self.n
        self.lambda_       = self.alpha**2 * (self.n + self.kappa) - self.n
        self.num_sigma_pts = 2*self.n + 1
        
        # Get the Weighting Matrices
        self.Get_Weight_Matrices() 
        
        # Perform Decorrelation
        # self.Perform_Decorrelation()
        
        # Generate the sigma points
        sigmaPoints = self.Generate_Sigma_Points()
        
        #######################################################################
        ###################### Apriori Update #################################
        
        # Compute the apriori output             
        aprioriOutput = self.PredictionStep(sigmaPoints)    
        
        # Unbox the apriori output
        aprioriMean   = aprioriOutput["mean"]
        aprioriCovar  = aprioriOutput["Covar"]
        aprioriPoints = aprioriOutput["aprioriPoints"] 
        
        #######################################################################
        ###################### Aposteriori Update #############################
            
        # Compute the aposteriori output
        aposterioriOutput = self.UpdateStep(aprioriPoints)
        
        # Unbox the aposteriori output
        aposterioriMean   = aposterioriOutput["mean"]
        aposterioriCovar  = aposterioriOutput["Covar"]
        aposterioriPoints = aposterioriOutput["aposterioriPoints"] 
        
        #######################################################################
        ######################### Residual Computation ########################
        
        # Compute residual from measurement
        yStar = self.y_k - aposterioriMean.reshape(-1,1)   
        
        #######################################################################
        ######################### UKF Gain Computation ########################
        
        # Prepare dictionary to compute cross covariance matrix  & UKF Gain
        funParam = {"input1": aprioriPoints, 
                    "input2": aposterioriPoints, 
                    "input1Mean": aprioriMean, 
                    "input2Mean": aposterioriMean, 
                    "weightMatrix": self.Wc}  
        
        uKFGain = self.Compute_UKF_Gain(funParam, aposterioriCovar)        
        
        #######################################################################
        ######################### UKF Estimate ################################
        
        # Compute Aposteriori State Update and Covariance Update
        x_hat  = aprioriMean.reshape(-1,1) + uKFGain @ yStar
        SigmaE = aprioriCovar - uKFGain @ aposterioriCovar @ uKFGain.T  
        
        # Prepare Output Dictionary
        x_hat = np.array([x_hat[0],
                          x_hat[1],
                          x_hat[2],
                          x_hat[3],
                          x_hat[4],
                          x_hat[5]])
        ukfOutput = {"x_hat": x_hat, "SigmaE": SigmaE}
        
        # Return UKF Estimate Output
        return ukfOutput     
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class NMPCParams:
    """
    Class to hold variables for NMPC Simulations
    """
    
    def __init__(self, sim_params):        
    
        # Simulation parameters
        self.T           = 0.1 # Discretization Time Steps
        self.len_horizon = config.carTrackHorizon # Prediction Horizon
        self.sim_time    = 10  # Total simulation time       
        
        # Dimensions of dynamics: Source: https://github.com/carla-simulator/carla/issues/135
        self.L           = 2.9 # Wheelbase of car (approx = 2.89 for Ford Mustang in Carla)  
        self.num_ctrls   = 2 # Number of controls
        self.num_states  = 6 # Number of states
        self.num_outputs = 4 # Number of outputs
        
        # State & Control Penalty Matrices
        self.Q = 100*np.eye(self.num_states) 
        self.Q[2,2] = 0
        self.R = 0.01*np.eye(self.num_ctrls)        
        
        # Convert from steering angles
        self._conv_rad_to_steer  = ((180.0/70.0)/np.pi)
        
        # Define the joint mean and covariance
        # Noise parameters
        self.mu_w = np.zeros(self.num_states)
        self.mu_v = np.zeros(self.num_outputs)                                         
        self.joint_dim = self.num_states + self.num_outputs
        self.joint_mu = np.append(self.mu_w, self.mu_v)

        # Unbox the noise data from pickle file
        # Unbox Pickle file to load the nodeList data 
        filename = 'noise_data.pkl'
        infile = open(filename,'rb')
        noise_data = pickle.load(infile)
        infile.close() 
        joint_Cov, WV = noise_data 
        
        
        # Generate independent Noises across all trials
        WV = np.random.multivariate_normal(self.joint_mu, joint_Cov, 5000).T
        
        # Set the covariance matrices for uncertain random variables
        self.joint_cov  = joint_Cov
        self.WV_Noise   = WV
        self.SigmaW     = self.joint_cov[0:self.num_states, 0:self.num_states]
        self.SigmaV     = self.joint_cov[self.num_states:, self.num_states:]
        self.CrossCorel = self.joint_cov[0:self.num_states, self.num_states:]                 
        self.SigmaE     = 0.01*np.eye(self.num_states) 
        
        # Constraints - Infer the boundary limits
        self.boundary  = sim_params["boundary"] # bottom_left [x,y, width, height] 
        self.obstacle  = sim_params["obstacle"]
        self.min_steer = -1.22 # minimum steering angle in radians = -35deg, -70deg = -1.22
        self.max_steer = 1.22  # maximum steering angle in radians = 35deg, 70 deg = 1.22
        self.a_max     = 8.0
        self.a_min     = 0# -8.0    
        self.v_min     = -15.0 # -15.0 #TODO make it 0   
        self.v_max     = 80.0  # 80.0
        self.x_min     = -85   # -300 # self.boundary[0] - 2.0 # -85.0 -2.0 = -87.0
        self.x_max     = -73   # 300 # self.boundary[0] + self.boundary[2] + 2.0 # -70.0 + 2.0 = -68.0       
        self.y_min     = 43    # 300 # self.boundary[1] # -8.0  SIGN CHANGED FOR TRACKING
        self.y_max     = 85    # -300 # self.boundary[1] + self.boundary[3] # -85.0 SIGN CHANGED FOR TRACKING
        self.obs_x_min = -80
        self.obs_x_max = -70
        self.obs_y_min = 40
        self.obs_y_max = 85        
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class NMPC_Tracker:
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
        self.nmpc_params = NMPCParams(sim_params)      
                
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
        setup_params["WV_Noise"] = self.nmpc_params.WV_Noise
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
        
        # Instantiate the NMPC controller for tracking waypoint        
        self.tracking_controller = NMPC_Steering(setup_params)
        
    ###############################################################################
    ###############################################################################
        
    
    def Update_State(self, robot_state, ob_state):
        """
        Updates the class variables using the robot_state and ob_state

        Parameters
        ----------
        robot_state : LIST OF FLOATS
            states of thr robot.
        ob_state : LIST OF FLOATS
            states of the obstacle.

        Returns
        -------
        None.

        """
        self.x = robot_state[0]
        self.y = robot_state[1]        
        self.yaw = robot_state[2]
        self.v = robot_state[3]
        self.ox = ob_state[0]
        self.oy = ob_state[1]
        
    ###############################################################################
    ###############################################################################
        
    def Update_Waypt(self, waypt):
        """
        Updates the waypoint information

        Parameters
        ----------
        waypt : CARLA Transform 
            Waypoint description.

        Returns
        -------
        None.

        """
        
        self.waypt_x   = waypt[0]
        self.waypt_y   = waypt[1]
        self.waypt_yaw = waypt[2]        
        self.waypt_v   = 0   
        
    ###############################################################################
    ###############################################################################
        
    @staticmethod
    def Bound_Angles(theta):
        """
        Converts angle from degrees to radians

        Parameters
        ----------
        theta : FLOAT
            Angle in degrees.

        Returns
        -------
        theta : FLOAT
            Angle in radians.

        """
        if theta < 0:
            theta = theta + 360
        if theta > 360:
            theta = theta - 360
        theta = theta*np.pi/180            
        return theta 
    
    ###############################################################################
    ###############################################################################

    def Get_Control_Inputs(self, ref_iterator, x, y, yaw, v, ox, oy, waypts):
        """
        Given a car player, obstacle & set of states, function Get_Control_Inputs
        applies the computed NMPC control inputs to steer the car player from 
        the current state to the commanded waypoint.        

        Parameters
        ----------
        x : FLOAT
            ego car x position.
        y : FLOAT
            ego car y position.
        yaw : FLOAT
            ego car yaw orientation.
        v : FLOAT
            ego car velocity.
        ox : FLOAT
            obstacle x position.
        oy : FLOAT
            obstacle y position.
        waypts : CARLA TRANSFORM OBJECTS
            contains the list of waypoints information.

        Returns
        -------
        None.

        """
        
        # Update the state variables
        self.Update_State([x, y, yaw, v], [ox, oy])
        
        # Select the last waypoint in the sequence
        waypt = waypts[:,-1]        
        
        # Update the waypoints
        self.Update_Waypt(waypt)
        
        # Prepare the start and next pts
        start_w = [self.x, self.y, self.yaw, self.v, self.ox, self.oy]
        next_w  = [self.waypt_x, self.waypt_y, self.waypt_yaw, self.waypt_v, self.ox, self.oy]
        
        # Obtain the Control Inputs by solving optimization problem       
        u, true_state, xhat = self.tracking_controller.NMPC_Track_Waypoint(ref_iterator, 
                                                     start_w, next_w, waypts,
                                                     self.nmpc_params.SigmaE) 

        return u, true_state, xhat                                         
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################

class Controller(object):
    def __init__(self):
        """
        Constructor function initilizing class variables

        Parameters
        ----------
        None.

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
        self._waypt_references   = None
        self._current_frame      = 0
        self._current_timestamp  = 0        
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
        sim_params["obstacle"]     = [-75, 55, 90, 2.9, 1.49]        
        sim_params["goal_x"]       = GOAL_POSITION[0]
        sim_params["goal_y"]       = GOAL_POSITION[1]
        sim_params["L"]            = 2.9                        
        sim_params["boundary"]     = [-83, 25, 13, 77] # bottom_left [x,y, width, height] SIGN CHANGED FOR TRACKING
        
        # Pass the dictionary parameters to initialize a MPC controller object
        # If tree build flag is true, instantiate a MPC class object
        # Else instantiate a NMPC_Tracker class object
        self.controller = NMPC_Tracker(sim_params) 
        
    ###########################################################################    
    def Update_Waypoint(self, waypt):
        """
        Update the states of robots & obstacles with timestampe and frame info

        Parameters
        ----------
        waypt : LIST OF FLOATS
            Reference states for N future time steps.

        Returns
        -------
        NONE

        """
        self._waypt_references = waypt
        
    ###########################################################################
    
    def Update_Information(self, robot_state, ob_state, timestamp):
        """
        Update the states of robots & obstacles with timestamp info

        Parameters
        ----------
        robot_state : LIST OF FLOATS
            Robot states.
        ob_state : LIST OF FLOATS
            Obstacle states.
        timestamp : FLOAT
            time stamp information.
        frame : FLOAT
            frame value.

        Returns
        -------
        _start_control_loop BOOLEAN
            variable encoding the readiness for tracking.

        """
        self._current_x         = robot_state[0]
        self._current_y         = robot_state[1]
        self._current_yaw       = robot_state[2]
        self._current_speed     = robot_state[3]
        self._obstacle_x        = ob_state[0]
        self._obstacle_y        = ob_state[1]
        self._current_timestamp = timestamp
        self._current_frame     = True
        
        # Update the state in controller
        self.controller.Update_State(robot_state, ob_state)
        
        # Decide to start the control loop or not
        if self._current_frame:
            return True
            
        return False    
    
    ###########################################################################
    ###########################################################################
    
    def Set_Control_Commands(self, input_throttle, input_steer_in_rad, input_brake):
        """
        Processes input commands to be within their respective ranges

        Parameters
        ----------
        input_throttle : FLOAT
            throlle value.
        input_steer_in_rad : FLOAT
            steering angle  value.
        input_brake : FLOAT
            braking value.

        Returns
        -------
        None.

        """
        
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)        
        
        # Convert radians to [- 1, 1] & clamp steering command to valid bounds        
        steer = np.fmax(np.fmin(self._conv_rad_to_steer * input_steer_in_rad, 1.0), -1.0)        
        
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)

        # Set the control commands
        self._set_steer    = steer     # in percent (0 to 1)
        self._set_brake    = brake     # in rad (-1.22 to 1.22)
        self._set_throttle = throttle  # in percent (0 to 1) 
    
    ###########################################################################
    ###########################################################################

    def Track_Waypoint(self, ref_iterator): 
        """
        Asks the ego player to to move to the commanded waypt by avoiding the 
        obstacle

        Parameters
        ----------
        waypt : Carla Transform
            contains the next waypoint position and orientation information

        Returns
        -------
        None.

        """
        
        # Get the states from the simulator
        x   = self._current_x
        y   = self._current_y
        yaw = self._current_yaw
        v   = self._current_speed
        ox  = self._obstacle_x
        oy  = self._obstacle_y   
        waypts = self._waypt_references

        # Get control inputs for tracking the current wayts
        u, true_state, xhat = self.controller.Get_Control_Inputs(ref_iterator, 
                                           x, y, yaw, v, ox, oy, waypts)
        
        return u, true_state, xhat
        
###############################################################################
###############################################################################
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
        min_y_obs = setup_params["obs_y_min"] # minimum obstacle y position - NEED TO CHANGE
        max_y_obs = setup_params["obs_y_max"] # maximum obstacle y position - NEED TO CHANGE
        
        
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
                                             constantObsVelocity])
        self.f_np = lambda x_, u_, w_: np.array([x_[3]*np.cos(x_[2]) + w_[0], 
                                                 x_[3]*np.sin(x_[2]) + w_[1], 
                                                 (x_[3]/self.L)*np.tan(u_[1]) + w_[2],
                                                 u_[0] + w_[3],
                                                 0,
                                                 constantObsVelocity])
        
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
        
        x_to_goal = newState[0,0] - self.goal_x
        y_to_goal = newState[1,0] - self.goal_y        
        output = [math.sqrt(x_to_goal**2 + y_to_goal**2), 
                  math.atan2(y_to_goal, x_to_goal) - newState[2,0],
                  newState[4,0]*np.cos(0.10),
                  newState[5,0]*np.sin(0.10)]
        
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
        
        steer_input = (1/np.deg2rad(45.0))*steer_angle # it was (1/1.22)*steer_angle
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
        # if accel > 0:
        #     brake_input    = 0.0
        #     throttle_input = accel    
        #     throttle_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
        # else:
        #     throttle_input = 0.0
        #     brake_input = accel    
        #     brake_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
        
        if accel > MAX_ACCEL:
            throttle_input = MAX_ACCEL
        else:
            throttle_input = accel
        brake_input = 0
            
        return throttle_input, brake_input
        
    
    ###############################################################################
    ###############################################################################
    
    def Prepare_Car_Controls(self, u_k):
        """
        Given a control input list, the function Prepare_Car_Controls prepares 
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
        
        throttle_input = u_k[0] # self.Get_Carla_Throttle_Input(u_k[0])
        steer_input = self.Get_Carla_Steer_Input(u_k[1])    
        car_controls = [throttle_input, steer_input]
        
        return car_controls
    
    ###############################################################################
    ###############################################################################
        
    def NMPC_Track_Waypoint(self, ref_iterator, start_w, next_w, waypts, SigmaE):
        """
        Given a starting and ending waypoints, NMPC_Track_Waypoint returns the 
        cost and steering result after applying NMPC based steering to the bicycle 
        model with state & input constraints. If no solution was found, a None 
        disctionary is returned. TO BE USED WHILE CAR IS TRACKING THE WAYPOINTS
        IN CARLA ENVIRONMENTS
    
        Parameters
        ----------
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
        #start_w[2] = pi_2_pi(start_w[2])
        #next_w[2] = pi_2_pi(next_w[2])
        
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
            print('Steering NLP Failed')
            return [], [], []
        
        # Obtain the optimal control input sequence
        u_sequence = sol.value(self.opt_controls)
        
        # Infer the first control input in that sequence
        u_k = u_sequence[0,:]        
        #u_k = u_sequence # Use this when N = 1
        
        # Get the predicted state trajectory for N time steps ahead
        next_states = sol.value(self.opt_states)                  
        
        # Generate the joint noise random variable           
        wv = self.WV_Noise[:, ref_iterator].reshape(-1,1)             
        # wv = npr.multivariate_normal(self.joint_mu, self.joint_cov).reshape(-1,1)         
        
        # Extract process noise and sensor noise from joint random variable
        w = wv[:self.num_states, :].reshape(-1,1) 
        v = wv[self.num_states:, :].reshape(-1,1) 
        w[4,0] = 0
        w[5,0] = 0

        # Prepare the carla control inputs
        car_controls = self.Prepare_Car_Controls(u_k)
        
        # Update the true nonlinear system
        x_gt   = true_state[0,0] + DT*true_state[3,0]*math.cos(true_state[2,0])
        y_gt   = true_state[1,0] + DT*true_state[3,0]*math.sin(true_state[2,0])
        yaw_gt = true_state[2,0] + DT*true_state[3,0]/LENGTH*math.tan(car_controls[1])
        v_gt   = true_state[3,0] + DT*car_controls[0]   
        obs_x  = true_state[4,0] + DT*0
        obs_y  = true_state[5,0] + DT*constantObsVelocity
        
        # Form the true state and add process noise
        true_state = np.array([x_gt, y_gt, yaw_gt, v_gt, obs_x, obs_y]).reshape(-1,1) + w
        
        # Create a measurement using the true state and the sensor noise
        output_no_noise = self.MeasurementModel(true_state)
        y_k = np.array(output_no_noise).reshape(-1,1) + v 
        
        # Update the estimator_params dictionary to call the UKF state estimator
        estimator_params["x_hat"]  = x_hat
        estimator_params["SigmaE"] = SigmaE  
        estimator_params["u_k"]    = u_k.T        
        estimator_params["y_k"]    = y_k
        
        # Call the Estimator Module to get the state estimate
        state_estimator  = State_Estimator(estimator_params)
        estimator_output = state_estimator.Get_Estimate()       
        
        # Unbox the UKF state estimate & covariance
        x_hat  = np.squeeze(estimator_output["x_hat"])
        SigmaE = estimator_output["SigmaE"]
        
        return car_controls, true_state, x_hat

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None
        
###############################################################################
###############################################################################


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

###############################################################################
###############################################################################

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
        plt.fill(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-r", alpha=0.5, label="Ego-Vehicle")
    else:
        plt.fill(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-k", alpha=0.5, label="Obstacle Vehicle")
    
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

def Do_Simulation(cx, cy, cyaw, dl, goal, initial_state, obstacle_state, controllerObject, timestamp):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    dl: course tick [m]
    """
    
    # Set time clock and reference iterator
    time = timestamp
    ref_iterator = 0
    
    # Infer goal, car state and obstacle state
    goal = GOAL_POSITION
    state = initial_state
    obsState = obstacle_state

    # Initialize state estimate
    xhat = np.array([state.x, state.y, state.yaw, state.v, obsState.x, obsState.y])
    
    # Initialize true state
    true_state = np.zeros((6,1))
    true_state[0,0] = state.x
    true_state[1,0] = state.y
    true_state[2,0] = state.yaw
    true_state[3,0] = state.v
    true_state[4,0] = obsState.x
    true_state[5,0] = obsState.y
    

    # Create placeholders for variables and initialize the lists
    t = [0.0] # Placeholder for time ticks
    d = [0.0] # Placeholder for steer angles
    a = [0.0] # Placeholder for acceleration
    x = [state.x] # Placeholder for true car x position
    y = [state.y] # Placeholder for true car y position
    v = [state.v] # Placeholder for true car velocity
    yaw  = [state.yaw] # Placeholder for true car yaw   
    obsx = [obsState.x] # Placeholder for obstacle car x position
    obsy = [obsState.y] # Placeholder for obstacle car y position
    xHats  = [xhat] # Placeholder for state estimate
    

    # Loop through the time until max time
    while time <= MAX_TIME:
        
        # Update the controllerObject with current states & simulation information
        robot_state = [true_state[0,0], true_state[1,0], true_state[2,0], true_state[3,0]]
        controllerObject.Update_Information(robot_state, [true_state[4,0], true_state[5,0]], time)
        
        # Get reference trajectory for next N time steps (N: planning horizon for MPC)
        horizon_ref_states = PATH_POINTS[:, ref_iterator:ref_iterator+config.carTrackHorizon+1]   # N = 10   
        
        # Update the waypoint information to the controller
        controllerObject.Update_Waypoint(horizon_ref_states)    
        
        # Compute the control inputs
        u, true_state, xhat = controllerObject.Track_Waypoint(ref_iterator) 
        
        # Infer the control inputs
        ai = u[0] # acceleration
        di = u[1] # steer
        
        # Update the time
        time = time + DT       

        # Update the history
        x.append(true_state[0,0])
        y.append(true_state[1,0])
        yaw.append(true_state[2,0])
        v.append(true_state[3,0])
        obsx.append(true_state[4,0])
        obsy.append(true_state[5,0])
        xHats.append(xhat)
        t.append(time)
        d.append(di)
        a.append(ai)
        
        # Increment the reference iterator
        ref_iterator = ref_iterator + 1

        # Stop the simulation if goal is reached
        if goal[0]-0.5 <= true_state[0,0] <= goal[0]+0.5 and goal[1]-0.5 <= true_state[1,0] <= goal[1]+0.5:
            print("Reached Goal")
            break

        if show_animation:  
            
            # Clear the axis
            plt.cla()
            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            # Plot the reference, obstacle and tracked trajectory
            plt.plot(obsx, obsy, "--k", label="Obstacle Vehicle Trajectory")
            plt.plot(cx, cy, "--r", label="Reference Trajectory")
            plt.plot(x, y, "--b", label="Ego vehicle Trajectory")
            
            # Plot the ego vehicle
            plot_car(true_state[0,0], true_state[1,0], true_state[2,0], steer=di, vehicleFlag=True)
            
            # Plot the obstacle vehicle
            plot_car(true_state[4,0], true_state[5,0], obstacle_state.yaw, steer=0, vehicleFlag=False)
            
            # Get the axis
            ax = plt.gca()
            # Plot goal region
            ax.add_artist(Rectangle((goal[0]-1, goal[1]-1), 2, 2 , lw=2, fc='b', ec='b', alpha = 0.3, label='Goal'))    
            # Plot start point
            ax.scatter(x[0], y[0], s=400, c='goldenrod', ec= 'k', linewidths=2, 
                       marker='^', label='Start Point', zorder=20, alpha = 0.3)
            
            # Set the Labels, titles and legends
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")            
            plt.axis("equal")
            plt.legend()
            plt.title("Time[s]:" + str(round(time, 2)) + ", speed[km/h]:" + str(round(true_state[3,0] * 3.6, 2)))
            plt.pause(0.00001)

    return t, x, y, yaw, v, d, a, obsx, obsy

###############################################################################
###############################################################################

def Get_Reference_Course(dl):
    
    # Extract x,y points from the path_points
    ax = PATH_POINTS[0,:]
    ay = PATH_POINTS[1,:]
    cyaw = PATH_POINTS[2,:] # [np.pi/2]*len(ax)

    return ax, ay, cyaw

###############################################################################
###############################################################################
########################### MAIN FUNCTION #####################################
###############################################################################
###############################################################################

def main():

    # course tick
    dl = 1.0  
    
    # Get the reference path
    cx, cy, cyaw, = Get_Reference_Course(dl)
    
    # Infer the goal position
    goal = GOAL_POSITION   

    # Get the initial vehicle state
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    
    # Get the initial obstalce vehicle state
    initial_obstacle_state = State(x=cx[0], y=cy[0]+10, yaw=cyaw[0], v=0.0)
    
    # Get the controller object and set it to MPC Controller
    controllerObject = Controller()
    
    # Update the controllerObject with current states & simulation information
    robot_state = [cx[0], cy[0], cyaw[0], 0]
    timestamp   = 0
    controllerObject.Update_Information(robot_state, [cx[0], cy[0]+10], timestamp)

    # Do the simulation
    t, x, y, yaw, v, d, a, obsx, obsy = Do_Simulation(cx, cy, cyaw, dl, goal,
                                                      initial_state, initial_obstacle_state,
                                                      controllerObject, timestamp)
    
    # Plot the final plots
    if show_plot:  
        # Close all existing plots
        plt.close("all")
        fig = plt.figure(figsize = [16,9])
        # create an axes object in the figure
        ax = fig.add_subplot(1, 1, 1)
        # plt.subplots()        
        ax.plot(cx, cy, "--r", label="reference trajectory")
        ax.plot(x, y, "-b", label="tracked trajectory")
        ax.plot(obsx, obsy, "--k", label="obstacle trajectory")
        ax.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        ax.legend()
        
        # Plot the ego vehicle
        plot_car(cx[0], cy[0], np.pi/2, steer=0, vehicleFlag=True)
        
        # Plot the obstacle vehicle start
        plot_car(cx[0], cy[0]+10, np.pi/2, steer=0, vehicleFlag=False)
        
        # Plot the obstacle vehicle end
        plot_car(obsx[-1], obsy[-1], np.pi/2, steer=0, vehicleFlag=False)
        
        
        # Plot goal region
        ax.add_artist(Rectangle((goal[0]-1, goal[1]-1), 2, 2 , lw=2, fc='b', ec='b', alpha = 0.3))    
        # Plot start point
        ax.scatter(cx[0], cy[0], s=400, c='goldenrod', ec= 'k', linewidths=2, 
                   marker='^', label='Start Point', zorder=20, alpha = 0.3)

        plt.show()

###############################################################################
###############################################################################

if __name__ == '__main__':
    main()