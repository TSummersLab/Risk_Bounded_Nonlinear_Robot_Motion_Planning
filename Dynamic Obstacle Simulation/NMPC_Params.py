# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:28:19 2021

@author: vxr131730
"""
###############################################################################
##############################################################################

import pickle
import numpy as np
from numpy import random as npr
from numpy import linalg as LA

###############################################################################
##############################################################################

class NMPCParams:
    """
    Class to hold variables for NMPC Simulations
    """
    
    def __init__(self, sim_params):        
    
        # Simulation parameters
        self.T           = 0.1 # Discretization Time Steps
        self.len_horizon = 50  # Prediction Horizon
        self.sim_time    = 10  # Total simulation time       
        
        # Dimensions of dynamics: Source: https://github.com/carla-simulator/carla/issues/135
        self.L           = 2.9 # Wheelbase of car (approx = 2.89 for Ford Mustang in Carla)  
        self.num_ctrls   = 2 # Number of controls
        self.num_states  = 6 # Number of states
        self.num_outputs = 4 # Number of outputs
        
        # State & Control Penalty Matrices
        self.Q = 10*np.eye(self.num_states) # Q = 1000*np.diag([0.1, 0.5, 0.1, 0.1]) # TODO SET 3RD ELEMENT OF Q TO ZERO
        self.Q[2,2] = 0
        self.R = 10*np.eye(self.num_ctrls)        
        
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
        
        # Prepare a joint covariance that is positive definite        
        # while True:
        #     joint_Cov = npr.rand(self.joint_dim,self.joint_dim)
        #     joint_Cov = 0.5*(joint_Cov + joint_Cov.T)
        #     joint_Cov = joint_Cov + (self.joint_dim*np.eye(self.joint_dim))
        #     joint_Cov = 0.0001*joint_Cov # 0.00001
        #     if np.all(np.linalg.eigvals(joint_Cov) > 0):
        #         break
        # print('Joint Covariance is', joint_Cov)
        self.joint_cov  = joint_Cov        
        self.SigmaW     = 0.0000001*np.eye(self.num_states) # self.joint_cov[0:self.num_states, 0:self.num_states]
        self.SigmaV     = 0.0000001*np.eye(self.num_outputs) # self.joint_cov[self.num_states:, self.num_states:]
        self.CrossCorel = np.zeros((self.num_states, self.num_outputs)) # self.joint_cov[0:self.num_states, self.num_states:]                 
        self.SigmaE     = 0.001*np.eye(self.num_states) # np.diag([0.05, 0.01, 0.01, 0.01, 0.001, 0.001])
        
        # Constraints - Infer the boundary limits
        self.boundary  = sim_params["boundary"] # bottom_left [x,y, width, height] 
        self.obstacle  = sim_params["obstacle"]
        self.min_steer = -1.22 # minimum steering angle in radians = -70deg = -1.22 rad
        self.max_steer = 1.22  # maximum steering angle in radians = 70 deg = 1.22 rad
        self.a_min     = -8.0 # -8.0    
        self.a_max     = 8.0 # 8.0        
        self.v_min     = 0 # -15.0 #TODO make it 0   
        self.v_max     = 80.0 # 80.0
        self.x_min     = -87 # -300 # self.boundary[0] - 2.0 # -85.0 -2.0 = -87.0
        self.y_min     = -85 #-300 # self.boundary[1] + self.boundary[3] # -85.0
        self.x_max     = -73 # 300 # self.boundary[0] + self.boundary[2] + 2.0 # = -73.0       
        self.y_max     = -43 # 300 # self.boundary[1] # -43.0  
        self.obs_x_min = self.obstacle[0]-1
        self.obs_y_min = self.obstacle[1]-1
        self.obs_x_max = self.obstacle[0]+1
        self.obs_y_max = self.obstacle[1]+1        
        
###############################################################################
###############################################################################