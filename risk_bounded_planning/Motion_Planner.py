#!/usr/bin/env python

###############################################################################
"""
@author: vxr131730 - Venkatraman Renganathan

This code simulates the bicycle dynamics of car by steering it using NMPC -
nonlinear model predictive control (multiple shooting technique) and the state 
estimation using (UKF) unscented kalman filter. This code uses CARLA simulator.

CARLA SIMULATOR VERSION - 0.9.10
PYTHON VERSION          - 3.6.8
VISUAL STUDIO VERSION   - 2017
UNREAL ENGINE VERSION   - 4.24.3

This script is tested in Python 3.6.8, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2020.  Email: vrengana@utdallas.edu

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
from casadi.tools import *
from numpy import linalg as LA


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

###############################################################################
####################### Define The Global Variable ############################
###############################################################################


IM_WIDTH   = 640         # Width of the simulation screen
IM_HEIGHT  = 480         # Height of the simulation screen
actor_list = []          # List of actors in the simulation world
l_r        = 1.415       # Distance from center of gravity to rear wheels
l_f        = 1.6         # Distance from center of gravity to front wheels
L          = l_r + l_f   # Total length of the vehicle
T          = 0.08        # Sampling Time (s)


IM_WIDTH            = 640
IM_HEIGHT           = 480 
red                 = carla.Color(255, 0, 0)
green               = carla.Color(0, 255, 0)
blue                = carla.Color(47, 210, 231)
cyan                = carla.Color(0, 255, 255)
yellow              = carla.Color(255, 255, 0)
orange              = carla.Color(255, 162, 0)
white               = carla.Color(255, 255, 255)
trail_life_time     = 50
waypoint_separation = 5


def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
    debug.draw_arrow(
        trans.location, trans.location + trans.get_forward_vector(),
        thickness=0.05, arrow_size=0.1, color=col, life_time=lt)


def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=5):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)

###############################################################################
###############################################################################
################## UNSCENTED KALMAN FILTER IMPLEMENTATION #####################
###############################################################################
###############################################################################

def UKF(ukf_parameters):
    
    # Unbox the input parameters
    zMean  = ukf_parameters["x_hat"] 
    u_k    = ukf_parameters["u_k"]     
    zCovar = ukf_parameters["SigmaE"]
    n_z    = ukf_parameters["n_z"]
    SigmaW = ukf_parameters["SigmaW"] 
    SigmaV = ukf_parameters["SigmaV"] 
    y_k    = ukf_parameters["y_k"] 
    
    # Define the global variables
    alpha         = 1.0
    beta          = 2.0
    n             = n_z
    kappa         = 10 - n
    lambda_       = alpha**2 * (n + kappa) - n
    num_sigma_pts = 2*n + 1
    
    # Initialize Van der Merwe's weighting matrix
    Wc = np.zeros((num_sigma_pts, 1))
    Wm = np.zeros((num_sigma_pts, 1))    
    
    # Compute the Van der Merwe's weighting matrix values    
    for i in range(num_sigma_pts):
        if i == 0:
            Wc[i,:] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
            Wm[i,:] = lambda_ / (n + lambda_)
            continue
        Wc[i,:] = 1/(2*(n + lambda_))
        Wm[i,:] = 1/(2*(n + lambda_))
       
    # Define the direction matrix
    U = LA.cholesky((n + lambda_)*zCovar)     
    
    # Generate the sigma points using Van der Merwe algorithm
    # Define Place holder for all sigma points
    sigmaPoints = np.zeros((n, num_sigma_pts))    
    
    # First SigmaPoint is always the mean 
    sigmaPoints[:,0] = zMean.T 
    
    # Generate sigmapoints symmetrically around the mean
    for k in range(n):
        sigmaPoints[:, k+1]   = sigmaPoints[:,0] + U[:, k]
        sigmaPoints[:, k+n+1] = sigmaPoints[:,0] - U[:, k]   
    
    ###################### Apriori Update #####################################
    # Compute the apriori output             
    aprioriOutput = PredictionStep(u_k, T, sigmaPoints, Wm, Wc, SigmaW)    
    
    # Unbox the apriori output
    aprioriMean   = aprioriOutput["mean"]
    aprioriCovar  = aprioriOutput["Covar"]
    aprioriPoints = aprioriOutput["aprioriPoints"] 
    
    ###########################################################################
    ###################### Aposteriori Update ###################Mean##############
        
    # Compute the aposteriori output
    aposterioriOutput = UpdateStep(aprioriPoints, Wm, Wc, SigmaV)
    
    # Unbox the aposteriori output
    aposterioriMean   = aposterioriOutput["mean"]
    aposterioriCovar  = aposterioriOutput["Covar"]
    aposterioriPoints = aposterioriOutput["aposterioriPoints"] 
    
    # Compute the residual yStar
    yStar = y_k - aposterioriMean.reshape(-1,1)   
     
    # Prepare dictionary to compute cross covariance matrix  
    funParam = {"input1": aprioriPoints, 
                "input2": aposterioriPoints, 
                "input1Mean": aprioriMean, 
                "input2Mean": aposterioriMean, 
                "weightMatrix": Wc}  
    
    # Compute the cross covariance matrix 
    crossCovarMatrix = ComputeCrossCovariance(funParam)
    
    # Compute Unscented Kalman Gain
    uKFGain = np.dot(crossCovarMatrix, LA.inv(aposterioriCovar))
    
    # Compute Aposteriori State Update and Covariance Update
    x_hat  = aprioriMean.reshape(-1,1) + uKFGain @ yStar
    SigmaE = aprioriCovar - uKFGain @ aposterioriCovar @ uKFGain.T  
    
    # Prepare Output Dictionary
    ukfOutput = {"x_hat": x_hat, "SigmaE": SigmaE}
    
    return ukfOutput 

###############################################################################

def PredictionStep(u_k, T, sigmaPoints, Wm, Wc, SigmaW):        
        
    # Get the shape of sigmaPoints
    ro, co = np.shape(sigmaPoints)
    # Create the data structure to hold the transformed points
    aprioriPoints = np.zeros((ro, co))
    
    # Loop through and pass each and every sigmapoint
    for i in range(co):
        aprioriPoints[:, i] = MotionModel(sigmaPoints[:, i], u_k, T)    
    
    # Compute the mean and covariance of the transformed points
    aprioriOutput = ComputeStatistics(aprioriPoints, Wm, Wc, SigmaW)
    
    # Add the aprioriPoints to output
    aprioriOutput["aprioriPoints"] = aprioriPoints 
    
    return aprioriOutput

###############################################################################

def UpdateStep(sigmaPoints, Wm, Wc, SigmaV):
    
    aprioriPoints = sigmaPoints 
    # Get the shape of aprioriPoints
    ro, M = np.shape(aprioriPoints)
    
    # Get the number of outputs
    num_outputs = SigmaV.shape[0]
       
    # Create the data structure to hold the transformed points
    aposterioriPoints = np.zeros((num_outputs, M)) #4 states, 2 outputs
    
    # Loop through and pass each and every sigmapoint
    for i in range(M):
        aposterioriPoints[:, i] = MeasurementModel(aprioriPoints[:, i])
    
    # Compute the mean and covariance of the transformed points    
    aposterioriOutput = ComputeStatistics(aposterioriPoints, Wm, Wc, SigmaV)
    
    # Add the aposterioriPoints to the output dictionary
    aposterioriOutput["aposterioriPoints"] = aposterioriPoints
    
    return aposterioriOutput

###############################################################################
def MotionModel(oldState, u, T):   
     
    newState = oldState + [T*oldState[3]*np.cos(oldState[2]), 
                           T*oldState[3]*np.sin(oldState[2]), 
                           T*(oldState[3]/L)*np.tan(u[1]*1.22),
                           T*u[0]*16]
    
    return newState

###############################################################################

def MeasurementModel(newState):    
    output = [math.sqrt(newState[0]**2 + newState[1]**2), 
              math.atan2(newState[1], newState[0])]
    
    return output

###############################################################################

def ComputeCrossCovariance(funParam):        
    
    # Compute the crossCovarMatrix    
    input1Shape = np.shape(funParam["input1"]) 
    input2Shape = np.shape(funParam["input2"])
    P           = np.zeros((input1Shape[0], input2Shape[0]))
    
    for k in range(input1Shape[1]):        
        diff1 =  funParam["input1"][:,k] - funParam["input1Mean"]
        diff2 =  funParam["input2"][:,k] - funParam["input2Mean"]       
        P     += funParam["weightMatrix"][k] * np.outer(diff1, diff2) 
    
    return P

###############################################################################

def ComputeStatistics(inputPoints, Wm, Wc, noiseCov):
    
    # Compute the weighted mean   
    inputPointsMean  = np.dot(Wm[:,0], inputPoints.T)
    
    # Compute the weighted covariance
    inputShape = np.shape(inputPoints)
    P          = np.zeros((inputShape[0], inputShape[0]))
    
    # Find the weighted covariance
    for k in range(inputShape[1]):        
        y = inputPoints[:, k] - inputPointsMean        
        P = P + Wc[k] * np.outer(y, y) 
    
    # Add the noise covariance
    P += noiseCov
    
    # Box the Output data
    statsOutput = {"mean": inputPointsMean, "Covar": P}
    
    return statsOutput

###############################################################################
###############################################################################

def TransformTheta(theta):
            if theta < 0:
                theta = 360 - abs(theta)
            return theta


###############################################################################
###############################################################################

def Get_Carla_Steer_Input(steer_angle):
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

def Get_Carla_Throttle_Input(accel):
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
    
    throttle_input = (1/3)*accel    
    throttle_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
    
    return throttle_input
    

###############################################################################
###############################################################################

def Prepare_CARLA_Controls(u_k):
    
    throttle_input = Get_Carla_Throttle_Input(u_k[0])
    steer_input    = Get_Carla_Steer_Input(u_k[1])    
    car_controls   = [throttle_input, steer_input]
    
    return car_controls

###############################################################################
###############################################################################
#######################  MAIN NMPC STEERING CODE ##############################
###############################################################################
###############################################################################

def Motion_Planner():
    
    # Define Simulation Parameters    
    N           = 30         # Prediction Horizon♥
    num_ctrls   = 2          # Number of controls
    num_states  = 4          # Number of states
    num_outputs = 2          # Number of outputs
    
    # CONTROL BOUNDS
    min_accel = -8    # minimum throttle
    max_accel = 8     # maximum throttle
    min_steer = -1.22 # minimum steering angle 
    max_steer = 1.22  # maximum steering angle    
    min_pos   = -300  # minimum position
    max_pos   = 300   # maximum position
    
    # Initiate an instance of opti class of casadi    
    opti = ca.Opti()
    
    # control variables, linear velocity v and angle velocity omega
    opt_controls = opti.variable(N, num_ctrls)
    opt_states   = opti.variable(N+1, num_states)
    accel        = opt_controls[:, 0]
    steer        = opt_controls[:, 1]    
    x            = opt_states[:, 0]
    y            = opt_states[:, 1]
    theta        = opt_states[:, 2]
    v            = opt_states[:, 3]
    
    # Define the State and Control Penalty matrices    
    Q = 0.1*np.diag([3,3,1,2])  # np.diag([3600,3600,1900,2]) 
    R = 0.1*np.diag([1,1]) # np.diag([1,8000]) 
    
    # Define the noise means
    mu_w = np.zeros(num_states)  # Mean of process noises
    mu_v = np.zeros(num_outputs) # Mean of sensor noises
    
    # Define Covariance Matrices
    SigmaW = np.diag([0.0005, 0.0005, 0, 0])  # Process Noise Covariance
    SigmaV = 0.0001*np.identity(num_outputs) # Sensor Noise Covariance
    SigmaE = 0.0001*np.identity(num_states)  # Estimation Error Covariance @t=0

    # parameters
    opt_x0 = opti.parameter(num_states)
    opt_xs = opti.parameter(num_states)
    
    # create model - Why *16, *1.22 ? Need to be answered
    f = lambda x_, u_: ca.vertcat(*[x_[3]*ca.cos(x_[2]), 
                                    x_[3]*ca.sin(x_[2]), 
                                    (x_[3]/L)*ca.tan(u_[1]*1.22),
                                    u_[0]*16])
    f_np = lambda x_, u_, w_: np.array([x_[3]*np.cos(x_[2]) + w_[0], 
                                        x_[3]*np.sin(x_[2]) + w_[1], 
                                        (x_[3]/L)*np.tan(u_[1]*1.22) + w_[2],
                                        u_[0]*16 + w_[3]])
    
    # Dynamics equality constraints - Denotes Multiple Shooting 
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i,:] + T*f(opt_states[i,:], opt_controls[i,:]).T
        opti.subject_to(opt_states[i+1,:] == x_next)

    # Define the cost function
    obj = 0 
    for i in range(N):
        obj += ca.mtimes([(opt_states[i, :] - opt_xs.T), Q, (opt_states[i, :] - opt_xs.T).T]) + \
               ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])

    # Set the objective to be minimized
    opti.minimize(obj)

    # Define the state and control constraints
    opti.subject_to(opti.bounded(min_pos, x, max_pos))
    opti.subject_to(opti.bounded(min_pos, y, max_pos))
    opti.subject_to(opti.bounded(-ca.inf, theta, ca.inf))
    opti.subject_to(opti.bounded(-15, v, 15))
    opti.subject_to(opti.bounded(min_accel, accel, max_accel))
    opti.subject_to(opti.bounded(min_steer, steer, max_steer))

    # Define the solver settings
    opts_setting = {'ipopt.max_iter':100, 
                    'ipopt.print_level':0, 
                    'print_time':0,   
                    'ipopt.acceptable_tol':1e-8, 
                    'ipopt.acceptable_obj_change_tol':1e-6}

    # Set the solver as IPOPT
    opti.solver('ipopt', opts_setting) 
    
    # Try running Carla - Create world, spawn actors & give waypoints to track    
    try:
        
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Once we have a client, retrieve the world that is currently running.
        world = client.get_world()
        
        # Get the debugger object
        debug = world.debug

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation. 
        blueprint_library = world.get_blueprint_library()

        # Choose Tesla model 3 car as the go vehicle
        ego_bp = blueprint_library.filter("model3")[0]
        
        # Define a starting pose for the car and plot it in the world 
        startpoint = world.get_map().get_spawn_points()[195] #128 195
        world.debug.draw_string(startpoint.location, 'O', draw_shadow = False, 
                                color = carla.Color(r=255, g=0, b=0), 
                                life_time = 50,
                                persistent_lines = True)
        
        # Spawn the car as actor in the world with its blueprint
        vehicle = world.spawn_actor(ego_bp, startpoint)
        
        # Append the actor to the list of actors
        actor_list.append(vehicle)
        
        
        # Record the starting location as the current waypoint
        current_w = world.get_map().get_waypoint(startpoint.location)
        
        # Number of waypoints
        num_waypoints = 1
        
        for k in range(num_waypoints):            
            # Record waypt separated by waypoint_separation as a potential waypt
            potential_w = list(current_w.next(waypoint_separation))
            
            # check for available right driving lanes
            if current_w.lane_change & carla.LaneChange.Right:
                right_w = current_w.get_right_lane()
                if right_w and right_w.lane_type == carla.LaneType.Driving:
                    potential_w += list(right_w.next(waypoint_separation))

            # check for available left driving lanes
            if current_w.lane_change & carla.LaneChange.Left:
                left_w = current_w.get_left_lane()
                if left_w and left_w.lane_type == carla.LaneType.Driving:
                    potential_w += list(left_w.next(waypoint_separation))

            # choose a random waypoint to be the next
            next_w = random.choice(potential_w)
            
            # Connect the waypoints using a line and mark the waypoints            
            draw_waypoint_union(debug, current_w, next_w, red, trail_life_time)
            draw_transform(debug, next_w.transform, white, trail_life_time)
        
            # update the current waypoint and sleep for some time
            current_w = next_w                    
        
        num_waypoints = 1
    
        #========== Start the NMPC Tracking Code ==============================                
        for waypt_num in range(num_waypoints):
            
            # Obtain new endpoint to be tracked
            # endpoint = waypoints[waypt_num].transform
            endpoint = next_w.transform
            
            
            # Connect it using string
            world.debug.draw_string(endpoint.location, 'O', 
                                    draw_shadow = False,
                                    color=carla.Color(r=0, g=0, b=255), 
                                    life_time=3, persistent_lines=True)
            
            # Update details of the next waypoint to be tracked using endpoint
            final_state = np.array([endpoint.location.x, 
                                    endpoint.location.y, 
                                    TransformTheta(startpoint.rotation.yaw),
                                    0.1])
            
            # Start from the current startpoint
            init_state  = np.array([startpoint.location.x, 
                                    startpoint.location.y, 
                                    TransformTheta(startpoint.rotation.yaw),
                                    0.1])
            
            # Initialize the true_state same as the initial_state
            true_state  = init_state.copy()    
            
            # Set the final state as opt_xs on opti structure
            opti.set_value(opt_xs, final_state)    
            
            # Define the initial control inputs sequence for all N time steps ahead   
            u_sequence = np.zeros((N, num_ctrls))
            
            # Create holders for sequence of next_states and current estimated state
            x_hat       = init_state.copy()  # Current estimated state            
            next_states = np.zeros((N+1, num_states)) # Sequence of next N+1 states
            
            # Set the UKF Parameter settings as a dictionary
            ukf_parameters = {"n_z": num_states,
                              "xTrue": true_state,                      
                              "SigmaW": SigmaW, 
                              "SigmaV": SigmaV, 
                              "x_hat": x_hat,
                              "u_k": u_sequence[0,:].T,
                              "SigmaE": SigmaE} 
            
            # Define place holders to store the history of control inputs and states
            u_k_hist = []
            x_k_hist = [x_hat] 
            S_k_hist = [SigmaE]
            x_true   = [true_state[0]]
            y_true   = [true_state[1]]
        
            # Start MPC Iteration
            sim_time     = 10.0       
            mpciter      = 0
            cost_total   = 0.0
            goal_toler   = 1.0     
            dist_to_goal = LA.norm(x_hat - final_state)
            
            while True:                
                
                # Current waypoint tracking is successful, track the next waypoint
                if dist_to_goal <= goal_toler or mpciter >= sim_time/T:
                    break
                
                # Set the parameter values - Update opt_x0 using x_hat
                opti.set_value(opt_x0, x_hat)
                
                # Set optimal control to the u_sequence
                opti.set_initial(opt_controls, u_sequence)# (N, 2)
                
                # Set the parameter opt_states to next_states
                opti.set_initial(opt_states, next_states) # (N+1, 3)
                
                # Solve the NMPC optimization problem
                sol = opti.solve()
                
                # Infer the total cost by evaluating the solved objective
                cost_total += sol.value(obj)
                
                # Obtain the optimal control input sequence
                u_sequence = sol.value(opt_controls)
                
                # Infer the first control input in that sequence
                u_k = u_sequence[0,:]
                
                # Get the predicted state trajectory for N time steps ahead
                next_states = sol.value(opt_states)   
                
                # Draw the predicted states
                for k in range(N):                       
                    world.debug.draw_string(carla.Location(x=float(next_states[k,0]), 
                                                           y=float(next_states[k,1]),
                                                           z=0.0), 
                                            'O', draw_shadow = False, 
                                            color=carla.Color(r=255, g=0, b=0), 
                                            life_time=0.01, persistent_lines=True)
                
                # Generate the process noise
                w = np.random.multivariate_normal(mu_w, SigmaW).reshape(-1,1)                
                
                # Prepare the carla control inputs
                car_controls = Prepare_CARLA_Controls(u_k)
                
                # Apply the control input to the carla simulator 
                vehicle.apply_control(carla.VehicleControl(throttle = float(car_controls[0]), 
                                                           steer    = float(car_controls[1])))
                
                # Wait for the next tick for the get_transform to work
                world.wait_for_tick()
                
                # Obtain where the car has landed in the simulator as true_state
                true_state = np.array([vehicle.get_transform().location.x, 
                                       vehicle.get_transform().location.y, 
                                       TransformTheta(vehicle.get_transform().rotation.yaw), 
                                       LA.norm([vehicle.get_velocity().x, vehicle.get_velocity().y])]).reshape(-1,1)
                
                true_state += w                
                
                # Generate the sensor noise
                v = np.random.multivariate_normal(mu_v, SigmaV).reshape(-1,1) 
                
                # Create a measurement using the true state and the sensor noise
                y_k = np.array(MeasurementModel(true_state)).reshape(-1,1) + v
                
                # Update the ukf_parameters dictionary to call the UKF state estimator
                ukf_parameters["x_hat"]  = x_hat
                ukf_parameters["u_k"]    = u_k.T
                ukf_parameters["SigmaE"] = SigmaE  
                ukf_parameters["y_k"]    = y_k
                
                # Call the UKF to get the state estimate
                ukf_output = UKF(ukf_parameters)
                
                # Unbox the UKF state estimate & covariance
                x_hat  = np.squeeze(ukf_output["x_hat"])
                SigmaE = ukf_output["SigmaE"]
                
                # Prepare u_sequence and next_states for next loop initialization
                # Since 1st entry is already used, trim it and repeat last entry to fill the vector
                # Eg. if u_sequence = [1 2 3 4 5 6], then fill as u_sequence = [2 3 4 5 6 6]
                # Why? u_sequence, next_states: best guesses for opt_controls, opt_states resp. in next iteration
                u_sequence  = np.concatenate((u_sequence[1:], u_sequence[-1:]))
                next_states = np.concatenate((next_states[1:], next_states[-1:]))
                
                # Store the history
                u_k_hist.append(u_k)
                x_k_hist.append(x_hat)
                S_k_hist.append(SigmaE)        
                x_true.append(true_state[0])
                y_true.append(true_state[1])
                
                # Update distance to goal
                dist_to_goal = np.linalg.norm(x_hat - final_state)
                
                # Increment the mpc loop counter
                mpciter = mpciter + 1   
                print('Iter Number:', mpciter)
            
        
        # Make sure the simulation stops and visible for few time steps
        time.sleep(10)
        
    finally:
        
        # Destroy all the actors if anything is spawned
    	for actor in actor_list:
    		actor.destroy()
    	print("All actors deleted")
    
