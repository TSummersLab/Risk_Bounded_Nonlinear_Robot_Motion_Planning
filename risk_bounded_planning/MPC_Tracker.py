#!/usr/bin/env python3
import pickle
import numpy as np
from numpy import random as npr
from numpy import linalg as LA
import Generate_Carla_Controls as Tracking_Controller

###############################################################################
##############################################################################

class NMPCParams:
    """
    Class to hold variables for NMPC Simulations
    """
    
    def __init__(self, sim_params):        
    
        # Simulation parameters
        self.T           = 0.1 # Discretization Time Steps
        self.len_horizon = 4  # Prediction Horizon
        self.sim_time    = 10  # Total simulation time       
        
        # Dimensions of dynamics: Source: https://github.com/carla-simulator/carla/issues/135
        self.L           = 2.9 # Wheelbase of car (approx = 2.89 for Ford Mustang in Carla)  
        self.num_ctrls   = 2 # Number of controls
        self.num_states  = 6 # Number of states
        self.num_outputs = 4 # Number of outputs
        
        # State & Control Penalty Matrices
        self.Q = 100*np.eye(self.num_states) # 100*np.diag([2, 2, 1, 5, 1, 1]) # Q = 1000*np.diag([0.1, 0.5, 0.1, 0.1]) # TODO SET 3RD ELEMENT OF Q TO ZERO
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
        WV = np.random.multivariate_normal(self.joint_mu, joint_Cov, 1000).T
        
        # # Prepare a joint covariance that is positive definite
        # while True:
        #     joint_Cov = npr.rand(self.joint_dim,self.joint_dim)
        #     joint_Cov = 0.5*(joint_Cov + joint_Cov.T)
        #     joint_Cov = joint_Cov + (self.joint_dim*np.eye(self.joint_dim))
        #     joint_Cov = 0.0001*joint_Cov
        #     if np.all(np.linalg.eigvals(joint_Cov) > 0):
        #         break
        self.joint_cov  = joint_Cov
        self.WV_Noise   = WV
        self.SigmaW     = self.joint_cov[0:self.num_states, 0:self.num_states]
        self.SigmaV     = self.joint_cov[self.num_states:, self.num_states:]
        self.CrossCorel = self.joint_cov[0:self.num_states, self.num_states:]                 
        self.SigmaE     = 0.01*np.eye(self.num_states) # np.diag([0.05, 0.01, 0.01, 0.01, 0.001, 0.001])
        
        # Constraints - Infer the boundary limits
        self.boundary  = sim_params["boundary"] # bottom_left [x,y, width, height] 
        self.obstacle  = sim_params["obstacle"]
        self.min_steer = -1.22 # minimum steering angle in radians = -35deg, -70deg = -1.22
        self.max_steer = 1.22  # maximum steering angle in radians = 35deg, 70 deg = 1.22
        self.a_max     = 8.0
        self.a_min     = -8.0    
        self.v_min     = -15.0 # -15.0 #TODO make it 0   
        self.v_max     = 80.0 # 80.0
        self.x_min     = -85# -300 # self.boundary[0] - 2.0 # -85.0 -2.0 = -87.0
        self.y_min     = -85# -300 # self.boundary[1] + self.boundary[3] # -85.0
        self.x_max     = -73# 300 # self.boundary[0] + self.boundary[2] + 2.0 # -70.0 + 2.0 = -68.0       
        self.y_max     = -43# 300 # self.boundary[1] # -8.0  
        self.obs_x_min = self.obstacle[0]-1
        self.obs_y_min = self.obstacle[1]-1
        self.obs_x_max = self.obstacle[0]+1
        self.obs_y_max = self.obstacle[1]+1        
        
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
        self.tracking_controller = Tracking_Controller.NMPC_Steering(setup_params)
        
    
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
        self.yaw = self.Bound_Angles(robot_state[2])
        self.v = robot_state[3]
        self.ox = ob_state[0]
        self.oy = ob_state[1]
        
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
        self.waypt_yaw = self.Bound_Angles(waypt[2])        
        self.waypt_v   = 0        
        
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
        theta = theta*np.pi/180            
        return theta 

    def Get_Control_Inputs(self, ego_player, static_player, ref_iterator, x, y, yaw, v, ox, oy, waypts):
        """
        Given a car player, obstacle & set of states, function Get_Control_Inputs
        applies the computed NMPC control inputs to steer the car player from 
        the current state to the commanded waypoint.        

        Parameters
        ----------
        ego_player : Carla Car Vehicle Object
            object representing the ego car in the CARLA environment.
        static_player : Carla Car Vehicle Object
            object representing the static obstacle car in the CARLA environment.
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
        self.tracking_controller.NMPC_Track_Waypoint(ego_player, static_player, ref_iterator, 
                                                     start_w, next_w, waypts,
                                                     self.nmpc_params.SigmaE)                                            
    
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
        sim_params["obstacle"]     = controller_params["obstacle"]        
        sim_params["goal_x"]       = controller_params["goal_x_pos"]
        sim_params["goal_y"]       = controller_params["goal_y_pos"]
        sim_params["L"]            = controller_params["wheelbase"]                        
        sim_params["boundary"]     = controller_params["boundary"] # bottom_left [x,y, width, height]
        
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
    
    def Update_Information(self, robot_state, ob_state, timestamp, frame):
        """
        Update the states of robots & obstacles with timestampe and frame info

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
        self._current_frame     = frame
        
        # Update the state in controller
        self.controller.Update_State(robot_state, ob_state)
        
        # Decide to start the control loop or not
        if self._current_frame:
            return True
            
        return False    
    
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

    def Track_Waypoint(self, ego_player, static_player, ref_iterator): 
        """
        Asks the ego player to to move to the commanded waypt by avoiding the 
        obstacle

        Parameters
        ----------
        ego_player : Carla Car Type Object
            object respresenting the ego car in the Carla environment.
        static_player : Carla Car Type Object
            object respresenting the obstacle car in the Carla environment.
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
        self.controller.Get_Control_Inputs(ego_player, static_player, ref_iterator, x, y, yaw, v, ox, oy, waypts)                    