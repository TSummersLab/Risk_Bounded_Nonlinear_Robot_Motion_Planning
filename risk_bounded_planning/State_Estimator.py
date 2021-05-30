# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:26:45 2020

@author: vxr131730

2D State Estimator Class to be used for the CARLA.
"""
###############################################################################
###############################################################################

import UKF_Estimator as UKF_Estimator 
import EKF_Estimator as EKF_Estimator 

###############################################################################
###############################################################################

class State_Estimator:
    
    def __init__(self, estimator_params):
        
        self.params    = estimator_params
        self.estimates = None
        
        # Plug in a UKF Estimator - Can also plugin a different estimator
        #self.estimator = UKF_Estimator.UKF() 
        self.estimator = EKF_Estimator.EKF() 

    ###########################################################################

    def Get_Estimate(self):
        
        estimates = self.estimator.Estimate(self.params)
        
        return estimates
    
###############################################################################
###############################################################################