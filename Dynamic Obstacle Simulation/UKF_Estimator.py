# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:29:47 2020

@author: vxr131730
"""

###############################################################################
####################### Import all the required libraries #####################
###############################################################################

import math
import numpy as np
from numpy import linalg as LA

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
                               0]
        
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
                  newState[4]*np.sin(0.10),
                  newState[5]*np.cos(0.10)]
        
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
        ukfOutput = {"x_hat": x_hat, "SigmaE": SigmaE}
        
        # Return UKF Estimate Output
        return ukfOutput     
    

###############################################################################
###############################################################################