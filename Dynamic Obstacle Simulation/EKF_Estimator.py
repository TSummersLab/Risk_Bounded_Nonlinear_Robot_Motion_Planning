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
################## EXTENDED KALMAN FILTER IMPLEMENTATION #####################
###############################################################################
###############################################################################

class EKF:
    
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
        
    def Get_System_Matrices(self):
        
        # Get equilibrium points
        xbar = self.zMean[0]
        ybar = self.zMean[1]
        thetabar = self.zMean[2]
        vbar = self.zMean[3]        
        deltabar = self.u_k[1]
        
        # Get linearization matrix terms
        xbardiff = xbar - self.goal_x
        ybardiff = ybar - self.goal_y
        diffsqrsum = xbardiff**2 + ybardiff**2
        sqrtdiffsqrsum = math.sqrt(diffsqrsum)
        
        # F matrix = df*/dx evaluated @ (\bar[x],\bar{u})
        # f* = f - H h(x), H is the pseudo gain
        self.F = np.eye(self.zMean.shape[0])
        
        # Populate F matrix entries        
        self.F[0,2] = -self.dT*vbar*math.sin(thetabar)
        self.F[0,3] = self.dT*math.cos(thetabar)                
        self.F[1,2] = self.dT*vbar*math.cos(thetabar)
        self.F[1,3] = self.dT*math.sin(thetabar)        
        self.F[2,3] = (self.dT/self.L)*math.tan(deltabar)     

        # # Populate F matrix entries
        # self.F = np.zeros((self.zMean.shape[0], self.zMean.shape[0]))
        # self.F[0,0] = 1 - self.dT*self.Tt*(xbardiff/sqrtdiffsqrsum)
        # self.F[0,1] = -self.dT*self.Tt*(ybardiff/sqrtdiffsqrsum)
        # self.F[0,2] = -self.dT*vbar*math.sin(thetabar)
        # self.F[0,3] = self.dT*math.cos(thetabar)
        # self.F[1,0] = -self.dT*self.Tt*(ybardiff/diffsqrsum)
        # self.F[1,1] = 1 - self.dT*self.Tt*(xbardiff/diffsqrsum)
        # self.F[1,2] = self.dT*vbar*math.cos(thetabar) + 1*self.Tt
        # self.F[1,3] = self.dT*math.sin(thetabar)
        # self.F[2,2] = 1
        # self.F[2,3] = (self.dT/self.L)*math.tan(deltabar)
        # self.F[2,4] = -self.Tt*math.cos(0.10)
        # self.F[3,3] = 1
        # self.F[3,5] = -self.Tt*math.cos(0.10)
            
    ###########################################################################
            
    def Perform_Decorrelation(self):        
        
        # Perform Decorrelation Scheme
        self.Tt = self.CrossCor @ LA.inv(self.SigmaV)    
        
        # Covariance of new process noise which has 0 cross-correlation with sensor noise
        self.SigmaW = self.SigmaW - self.Tt @ self.SigmaV @ self.Tt.T        
                
        # Fabricate the known input
        self.knownIP = np.squeeze(self.Tt @ self.y_k)       
                    
    ###########################################################################    
    
    def PredictionStep(self):        
            
        self.aprioriMean = self.NewMotionModel(self.zMean)  
        self.aprioriCovar = self.F @ self.zCovar @ self.F.T + self.SigmaW
    
    ###########################################################################
    
    def UpdateStep(self):    
        
        # Get equilibrium points
        xbar = self.zMean[0]
        ybar = self.zMean[1]        
        
        # Get linearization matrix terms
        xbardiff = xbar - self.goal_x
        ybardiff = ybar - self.goal_y
        diffsqrsum = xbardiff**2 + ybardiff**2
        sqrtdiffsqrsum = math.sqrt(diffsqrsum)
        
        # H matrix = dh/dx evaluated @ (\bar[x],\bar{u})        
        self.H = np.zeros((self.SigmaV.shape[0], self.zMean.shape[0]))        

        # Populate H matrix entries
        self.H[0,0] = self.dT*(xbardiff/sqrtdiffsqrsum)
        self.H[0,1] = self.dT*(ybardiff/sqrtdiffsqrsum)        
        self.H[1,0] = self.dT*(ybardiff/diffsqrsum)
        self.H[1,1] = self.dT*(xbardiff/diffsqrsum)
        self.H[1,2] = -1                
        self.H[2,4] = math.cos(0.10)        
        self.H[3,5] = math.sin(0.10)
        
        # Compute Aposteriori Mean and Covariance Matrix
        self.aposterioriMean = np.array(self.MeasurementModel(self.aprioriMean), dtype=object)
        self.aposterioriCovar = self.H @ self.aprioriCovar @ self.H.T + self.SigmaV
    
    ###########################################################################
    def NewMotionModel(self, oldState):
        
        newState = self.MotionModel(oldState) - self.Tt @ self.MeasurementModel(oldState) + self.knownIP        
        
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
                  newState[4]*np.cos(0.10),
                  newState[5]*np.sin(0.10)]
        
        return output
        
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
        
        # Perform Decorrelation
        self.Perform_Decorrelation()
        
        # Get the System Matrices
        self.Get_System_Matrices() 
        
        #######################################################################
        ###################### Apriori Update #################################
        
        # Compute the apriori output             
        self.PredictionStep()             
        
        #######################################################################
        ###################### Aposteriori Update #############################
            
        # Compute the aposteriori output
        self.UpdateStep()
        
        #######################################################################
        ######################### Residual Computation ########################
        
        # Compute residual from measurement
        self.yResidual = self.y_k - self.aposterioriMean.reshape(-1,1)        
        
        #######################################################################
        ######################### EKF Gain Computation ########################
        
        self.eKFGain = self.aprioriCovar @ self.H.T @ LA.inv(self.aposterioriCovar)
        
        #######################################################################
        ######################### EKF Estimate ################################
        
        # Compute Aposteriori State Update and Covariance Update
        x_hat  = self.aprioriMean.reshape(-1,1) + self.eKFGain @ self.yResidual
        SigmaE = self.aprioriCovar - self.eKFGain @ self.H @ self.aprioriCovar
        
        # Prepare Output Dictionary
        ekfOutput = {"x_hat": x_hat, 
                     "SigmaE": SigmaE}
        
        # Return UKF Estimate Output
        return ekfOutput    

###############################################################################
###############################################################################