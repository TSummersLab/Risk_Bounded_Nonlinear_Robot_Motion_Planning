#!/usr/bin/env python3
"""
Changelog:
New is v1_1:
- Fixes bug in heading angle when rewiring the tree

New is v1_0:
- Run DR-RRT* with unicycle dynamics for steering

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Venkatraman Renganathan
Email:
vrengana@utdallas.edu
Date created: Mon Jan 13 12:10:03 2020
Contributions: DR-RRT* base code (structures, expanding functions, rewire functions, and plotting)
(C) Venkatraman Renganathan, 2019.

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
Github:
@The-SS
Contributions: Different steering function, updated steering and rewiring functions, adding features, DR-RRT* updates
Note: with this update, all DR checks have been replaced with normal checks

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script performs RRT or RRT* path planning using the unicycle dynamics for steering. Steering is achieved by solving
a nonlinear program (NLP) using Casadi.

Tested platform:
- Python 3.6.9 on Ubuntu 18.04 LTS (64 bit)


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

###############################################################################
###############################################################################

# Import all the required libraries
import random
import math
import csv
import numpy as np
import numpy.random as npr
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import sys
import pickle
from os import system
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.collections import EllipseCollection
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from casadi import *
from casadi.tools import *
from scipy.linalg import block_diag
from scipy.special import erfinv
from numpy.linalg import inv
from numpy import linalg as LA
from namedlist import namedlist
import copy
import os
import UKF_Estimator as UKF_Estimator
from scripts.plotting import plot_env

np.seterr(divide='ignore')
###############################################################################
###############################################################################

# Defining Global Variables (User chosen)
# See config file for defaults
import config
NUMSAMPLES = config.NUMSAMPLES  # total number of samples
STEER_TIME = config.STEER_TIME  # Maximum Steering Time Horizon
ENVCONSTANT = config.ENVCONSTANT  # Environment Constant for computing search radius
DT = config.DT  # timestep between controls
GOALAREA = copy.deepcopy(config.GOALAREA)  # [xmin,xmax,ymin,ymax] Goal zone
ROBSTART = config.ROBSTART  # robot starting location (x,y)
RANDAREA = copy.deepcopy(config.RANDAREA)  # area sampled: [xmin,xmax,ymin,ymax], [-4.7, 4.7, -4.7, 4.7] good with 0 ROBRAD, limit:[-5,5,-5,5]
RRT = config.RRT  # True --> RRT, False --> RRT*
DRRRT = config.DRRRT # True --> apply DR checks, False --> regular RRT
MAXDECENTREWIRE =  config.MAXDECENTREWIRE  # maximum number of descendents to rewire
RANDNODES = config.RANDNODES  # false --> only 5 handpicked nodes for debugging
SATLIM = config.SATLIM  # saturation limit (random nodes sampled will be cropped down to meet this limit from the nearest node)
ROBRAD = config.ROBRAD  # radius of robot (added as padding to environment bounds and the obstacles
SBSP = config.SBSP  # Shrinking Ball Sampling Percentage (% nodes in ball to try to rewire) (100 --> all nodes rewired)
SBSPAT = config.SBSPAT  # SBSP Activation Threshold (min number of nodes needed to be in the shrinking ball for this to activate)
SAVEDATA = config.SAVEDATA # True --> save data, False --> don't save data
SAVEPATH = config.SAVEPATH  # path to save data
OBSTACLELIST = copy.deepcopy(config.OBSTACLELIST)  # [ox,oy,wd,ht]
SIGMAW = config.SIGMAW  # Covariance of process noise
SIGMAV = config.SIGMAV  # Covariance of sensor noise (we don't have any for now)
CROSSCOR = config.CROSSCOR   # Cross Correlation between the two noises (none for now)
ALFA = config.ALFA  # risk bound
QHL = config.QHL  # Q matrix for quadratic cost
RHL = config.RHL  # R matrix for quadratic cost
# To edit robot speed go to `GetDynamics`
# To edit input and state quadratic cost matrices Q and R go to `SetUpSteeringLawParameters`

# Defining Global Variables (NOT User chosen)
import file_version
FILEVERSION = file_version.FILEVERSION  # version of this file
SAVETIME = str(int(time.time()))  # Used in filename when saving data
if not RANDNODES:  # if
    NUMSAMPLES = 5
try:  # try to create save directory if it doesn't exist
    os.mkdir(SAVEPATH)
except:
    print('Save directory exists')


# Define the namedlists
Dynamics = namedlist("Dynamics", "numStates numControls")
StartParams = namedlist("StartParams",
                        "start randArea goal1Area maxIter plotFrequency obstacleList dynamicsData")
SteerSetParams = namedlist("SteerSetParams", "dt f N solver argums numStates numControls")
SteerSetParams2 = namedlist("SteerSetParams2", "dt f N solver argums numStates numControls")

###############################################################################
###############################################################################

class trajNode():
    """
    Class Representing a steering law trajectory Node
    """

    def __init__(self, numStates, numControls):
        """
        Constructor Function
        """
        self.X = np.zeros((numStates, 1))  # State Vector
        self.Sigma = np.zeros((numStates, numStates, 1))  # Covariance Matrix
        self.Ctrl = np.zeros((numControls))  # Control at trajectory node


###############################################################################
###############################################################################

class DR_RRTStar_Node():
    """
    Class Representing a DR-RRT* Tree Node
    """

    def __init__(self, numStates, numControls, num_traj_nodes):
        """
        Constructor Function
        """
        self.cost = 0.0  # Cost
        self.parent = None  # Index of the parent node
        self.means = np.zeros((num_traj_nodes, numStates, 1))  # Mean Sequence
        self.covars = np.zeros((num_traj_nodes, numStates, numStates))  # Covariance matrix sequence
        self.inputCommands = np.zeros((num_traj_nodes - 1, numControls))  # Input Commands to steer from parent to the node itself

    ###########################################################################

    def __eq__(self, other):
        """
        Overwriting equality check function to compare two same class objects
        """
        costFlag = self.cost == other.cost
        parentFlag = self.parent == other.parent
        meansFlag = np.array_equal(self.means, other.means)
        covarsFlag = np.array_equal(self.covars, other.covars)

        return costFlag and parentFlag and meansFlag and covarsFlag

    ###############################################################################


###############################################################################
class DR_RRTStar():
    """
    Class for DR-RRT* Planning
    """

    def __init__(self, startParam):
        """
        Constructor function
        Input Parameters:
        start   : Start Position [x,y]
        randArea: Ramdom Samping Area [xmin,xmax,ymin,ymax]
        goalArea: Goal Area [xmin,xmax,ymin,ymax]
        maxIter : Maximum # of iterations to run for constructing DR-RRT* Tree
        """
        # Unwrap the StartParameters
        start, randArea, goal1Area, maxIter, plotFrequency, obstacleList, Dynamics = startParam

        # Add the Double Integrator Data
        self.iter = 0
        self.controlPenalty = 0.02
        self.plotFrequency = plotFrequency
        self.xminrand = randArea[0]
        self.xmaxrand = randArea[1]
        self.yminrand = randArea[2]
        self.ymaxrand = randArea[3]
        self.x1mingoal = goal1Area[0]
        self.x1maxgoal = goal1Area[1]
        self.y1mingoal = goal1Area[2]
        self.y1maxgoal = goal1Area[3]
        self.maxIter = maxIter
        self.obstacleList = obstacleList
        self.dynamicsData = Dynamics
        self.numStates, self.numControls = self.dynamicsData
        self.alfaThreshold = 0.01 #0.05
        self.alfa = [self.alfaThreshold] * len(self.obstacleList)
        self.alfa = copy.deepcopy(ALFA)
        self.saturation_limit = SATLIM
        self.R = []
        self.Q = []
        self.dropped_samples = []
        self.S0 = np.array([[0.1, 0, 0],
                           [0, 0.1, 0],
                           [0, 0, 0.1]])
        # Define the covariances of process and sensor noises
        self.numOutputs = self.numStates
        self.SigmaW = SIGMAW  # 0.005 is good # Covariance of process noise
        self.SigmaV = SIGMAV  # Covariance of sensor noise
        self.CrossCor = CROSSCOR # Cross Correlation between the two noises. Remove this later

        # Initialize DR-RRT* tree node with start coordinates
        self.InitializeTree(start)

        ###########################################################################

    def InitializeTree(self, start):
        """
        Prepares DR-RRT* tree node with start coordinates & adds to nodeList
        """

        # Unwrap the dynamicsData
        self.numStates, self.numControls = self.dynamicsData

        # Create an instance of DR_RRTStar_Node class
        num_traj_nodes = 1
        self.start = DR_RRTStar_Node(self.numStates, self.numControls, num_traj_nodes)

        for k in range(num_traj_nodes):
            self.start.means[k, 0, :] = start[0]
            self.start.means[k, 1, :] = start[1]
            self.start.covars[k, :, :] = self.S0
            # heading is already initialized to zero # TODO: try setting it to point to goal
            # no need to update the input since they are all zeros as initialized

            # Add the created start node to the nodeList
        self.nodeList = [self.start]

        ###########################################################################

    def GetDynamics(self):

        ###########################################################################
        ######################### Begin Modular Part ##############################
        ########## Can be changed representing any nonlinear system dynamics ######
        ###########################################################################

        # Define steer function variables
        dt = DT  # discretized time step
        N = STEER_TIME  # Prediction horizon

        # Unwrap the dynamicsData
        numStates, numControls = self.dynamicsData

        # Define symbolic states using Casadi SX
        x = SX.sym('x')  # x position
        y = SX.sym('y')  # y position
        theta = SX.sym('theta')  # heading angle
        states = vertcat(x, y, theta)  # all three states

        # Define symbolic inputs using Cadadi SX
        v = SX.sym('v')  # linear velocity
        omega = SX.sym('omega')  # angular velocity
        controls = vertcat(v, omega)  # both controls

        # RHS of nonlinear unicycle dynamics (continuous time model)
        rhs = horzcat(v * cos(theta), v * sin(theta), omega)

        # Nonlinear State Update function f(x,u)
        # Given {states, controls} as inputs, returns {rhs} as output
        f = Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

        # Input and state constraints
        v_max = 0.5  # maximum linear velocity (m/s)
        v_min = - v_max  # minimum linear velocity (m/s)
        omega_max = np.pi  # 0.125 * (2 * np.pi)  # maximum angular velocity (rad/s)
        omega_min = -omega_max  # minimum angular velocity (rad/s)
        x_max = 5  # maximum state in the horizontal direction
        x_min = -5  # minimum state in the horizontal direction
        y_max = 5  # maximum state in the vertical direction
        y_min = -5  # minimum state in the vertical direction
        theta_max = np.inf  # maximum state in the theta direction
        theta_min = -np.inf  # minimum state in the theta direction

        lbx = []
        ubx = []
        lbg = 0.0
        ubg = 0.0
        # Upper and lower bounds on controls
        for _ in range(N):
            lbx.append(v_min)
            ubx.append(v_max)
        for _ in range(N):
            lbx.append(omega_min)
            ubx.append(omega_max)
        # Upper and lower bounds on states
        for _ in range(N + 1):
            lbx.append(x_min)
            ubx.append(x_max)
        for _ in range(N + 1):
            lbx.append(y_min)
            ubx.append(y_max)
        for _ in range(N + 1):
            lbx.append(theta_min)
            ubx.append(theta_max)

        # Create the arguments dictionary to hold the constraint values
        argums = {'lbg': lbg,
                  'ubg': ubg,
                  'lbx': lbx,
                  'ubx': ubx}

        return dt, N, argums, f

    ###########################################################################

    def ComputeL2Distance(self, fromNode, toNode):
        """
        Returns the distance between two nodes computed using the Euclidean distance metric
        Input parameters:
        fromNode   : Node representing point A
        toNode     : Node representing point B
        """

        # Use the dynamic control-based distance metric
        diffVec = (fromNode.means[-1, :, :] - toNode.means[-1, :, :])[0:2, 0]  # no heading

        # Compute the radial (Euclidean) target distance
        dist = LA.norm(diffVec)
        return dist

    def SaturateNodeWithL2(self, fromNode, toNode):
        '''
        If the L2 norm of ||toNode-fromNode|| is greater than some saturation distance, find a new node newToNode along
        the vector (toNode-fromNode) such that ||newToNode-fromNode|| = saturation distance
        Inputs:
        fromNode: from/source node (type: DR_RRTStar_Node)
        toNode: desired destination node (type: DR_RRTStar_Node)
        Oupputs:
        newToNode: either the original toNode or a new one (type: DR_RRTStar_Node)
        '''
        if self.ComputeL2Distance(fromNode,
                                  toNode) > self.saturation_limit:  # node is further than the saturation distance
            from_x = fromNode.means[-1, :, :][0, 0]
            from_y = fromNode.means[-1, :, :][1, 0]
            to_x = toNode.means[-1, :, :][0, 0]
            to_y = toNode.means[-1, :, :][1, 0]
            angle = math.atan2(to_y - from_y, to_x - from_x)
            new_to_x = from_x + self.saturation_limit * math.cos(angle)
            new_to_y = from_y + self.saturation_limit * math.sin(angle)
            newToNode = toNode
            newToNode.means[-1, :, :][0, 0] = new_to_x
            newToNode.means[-1, :, :][1, 0] = new_to_y
            return newToNode
        else:
            return toNode

    ###########################################################################

    def RandFreeChecks(self, x, y):
        """
        Performs Collision Check For Random Sampled Point
        Inputs:
        x,y : Position data which has to be checked for collision
        Outputs:
        True if safe, False if collision
        """

        for ox, oy, wd, ht in self.obstacleList:
            if (x >= ox and x <= ox + wd and y >= oy and y <= oy + ht):
                return False  # collision
        return True  # safe

    ###########################################################################

    def GetRandomPoint(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """
        # Create a randNode as a tree object
        num_traj_nodes = 1
        randNode = DR_RRTStar_Node(self.numStates, self.numControls, num_traj_nodes)

        # Initialize using the generated free points
        xFreePoints, yFreePoints, thetaFreePoints = self.freePoints
        randNode.means[-1, 0, :] = xFreePoints[self.iter]
        randNode.means[-1, 1, :] = yFreePoints[self.iter]
        randNode.means[-1, 2, :] = thetaFreePoints[self.iter]

        return randNode

        ###########################################################################

    def GetFreeRandomPoints(self):

        xFreePoints = []
        yFreePoints = []
        thetaFreePoints = []

        if RANDNODES:
            # get 3% of the sampled points from the goal. The number is bounded within 2 and 10 nodes (2 < 3% < 10)
            # (added at the end of the list of nodes)
            points_in_goal = min(self.maxIter, min(max(2, int(3 / 100 * self.maxIter)), 10))
            for iter in range(self.maxIter - points_in_goal):
                # Sample uniformly around sample space
                searchFlag = True
                while searchFlag:
                    # initialize with a random position in space with orientation = 0
                    xPt = random.uniform(self.xminrand, self.xmaxrand)
                    yPt = random.uniform(self.yminrand, self.ymaxrand)
                    if self.RandFreeChecks(xPt, yPt):
                        break
                xFreePoints.append(xPt)
                yFreePoints.append(yPt)
                thetaFreePoints.append(random.uniform(-np.pi, np.pi))
            for iter in range(points_in_goal):
                xPt = random.uniform(self.x1mingoal, self.x1maxgoal)
                yPt = random.uniform(self.y1mingoal, self.y1maxgoal)
                thetaPt = random.uniform(-np.pi, np.pi)
                xFreePoints.append(xPt)
                yFreePoints.append(yPt)
                thetaFreePoints.append(thetaPt)
        else:
            # pre-chosen nodes (for debugging only)
            xFreePoints.append(-3)
            yFreePoints.append(-4)
            thetaFreePoints.append(0)
            xFreePoints.append(-1)
            yFreePoints.append(-4)
            thetaFreePoints.append(0)
            xFreePoints.append(0.5)
            yFreePoints.append(-4)
            thetaFreePoints.append(np.pi / 2)
            xFreePoints.append(0.5)
            yFreePoints.append(0.5)
            thetaFreePoints.append(np.pi / 2)
            xFreePoints.append(0)
            yFreePoints.append(-1.5)
            thetaFreePoints.append(np.pi / 4)

        self.freePoints = [xFreePoints, yFreePoints, thetaFreePoints]

    ###########################################################################

    def GetAncestors(self, childNode):
        """
        Returns the complete list of ancestors for a given child Node
        """
        ancestorNodeList = []
        while True:
            if childNode.parent is None:
                # It is root node - with no parents
                ancestorNodeList.append(childNode)
                break
            elif childNode.parent is not None:
                ancestorNodeList.append(self.nodeList[childNode.parent])
                childNode = self.nodeList[childNode.parent]
        return ancestorNodeList

        ###########################################################################

    def GetNearestListIndex(self, randNode):
        """
        Returns the index of the node in the tree that is closest to the randomly sampled node
        Input Parameters:
        randNode  : The randomly sampled node around which a nearest node in the DR-RRT* tree has to be returned
        """
        distanceList = []
        for node in self.nodeList:
            distanceList.append(self.ComputeL2Distance(node, randNode))
        return distanceList.index(min(distanceList))

    ###########################################################################

    def PrepareTrajectory(self, meanValues, covarValues, inputCommands):
        """
        Prepares the trajectory as trajNode from steer function outputs

        Input Parameters:
        meanValues : List of mean values
        covarValues : List of covariance values
        inputCommands: List of input commands

        Output Parameters:
        xTrajs: List of TrajNodes
        """

        T = len(inputCommands)
        # Trajectory data as trajNode object for each steer time step
        xTrajs = [trajNode(self.numStates, self.numControls) for i in range(T + 1)]
        for k, xTraj in enumerate(xTrajs):
            xTraj.X = meanValues[k]
            xTraj.Sigma = covarValues[k] # TODO: CHECK DIM'N/ELEMENT ACCESS
            if k < T:
                xTraj.Ctrl = inputCommands[k]
        return xTrajs

    ###########################################################################

    def SetUpSteeringLawParameters(self):

        # Unwrap the dynamicsData
        numStates, numControls = self.dynamicsData

        # Get the dynamics specific data
        dt, N, argums, f = self.GetDynamics()

        # Define state and input cost matrices for solving the NLP
        Q = QHL
        R = RHL
        self.Q = Q
        self.R = R

        # Casadi SX trajectory variables/parameters for multiple shooting
        U = SX.sym('U', N, numControls)  # N trajectory controls
        X = SX.sym('X', N + 1, numStates)  # N+1 trajectory states
        P = SX.sym('P', numStates + numStates)  # first and last states as independent parameters

        # Concatinate the decision variables (inputs and states)
        opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

        # Cost function
        obj = 0  # objective/cost
        g = []  # equality constraints
        g.append(X[0, :].T - P[:3])  # add constraint on initial state
        for i in range(N):
            # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
            obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
            # compute the next state from the dynamics
            x_next_ = f(X[i, :], U[i, :]) * dt + X[i, :]
            # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
            g.append(X[i + 1, :].T - x_next_.T)
        g.append(X[N, 0:2].T - P[3:5])  # constraint on final state

        # Set the nlp problem
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

        # Set the nlp problem settings
        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,  # 4
                        'print_time': 0,
                        'verbose': 0,  # 1
                        'error_on_fail': 1}

        # Create a solver that uses IPOPT with above solver settings
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.SteerSetParams = SteerSetParams(dt, f, N, solver, argums, numStates, numControls)

    def SetUpSteeringLawParametersWithFinalHeading(self):
        '''
        Same as SetUpSteeringLawParameters but with all three states enforced at the end instead of only x-y states
        This is used when rewiring and updating descendants
        '''
        # Unwrap the dynamicsData
        numStates, numControls = self.dynamicsData

        # Get the dynamics specific data
        dt, N, argums, f = self.GetDynamics()

        # Define state and input cost matrices for solving the NLP
        Q = QHL
        R = RHL
        self.Q = Q
        self.R = R

        # Casadi SX trajectory variables/parameters for multiple shooting
        U = SX.sym('U', N, numControls)  # N trajectory controls
        X = SX.sym('X', N + 1, numStates)  # N+1 trajectory states
        P = SX.sym('P', numStates + numStates)  # first and last states as independent parameters

        # Concatinate the decision variables (inputs and states)
        opt_variables = vertcat(reshape(U, -1, 1), reshape(X, -1, 1))

        # Cost function
        obj = 0  # objective/cost
        g = []  # equality constraints
        g.append(X[0, :].T - P[:3])  # add constraint on initial state
        for i in range(N):
            # add to the cost the quadratic stage cost: (x-x_des)*Q*(x-x_des)^T + u*R*u^T
            obj += mtimes([U[i, :], R, U[i, :].T])  # quadratic penalty on control effort
            # compute the next state from the dynamics
            x_next_ = f(X[i, :], U[i, :]) * dt + X[i, :]
            # make the dynamics' next state the same as the i+1 trajectory state (multiple shooting)
            g.append(X[i + 1, :].T - x_next_.T)
        g.append(X[N, 0:3].T - P[3:])  # constraint on final state including the heading angle

        # Set the nlp problem
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': vertcat(*g)}

        # Set the nlp problem settings
        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,  # 4
                        'print_time': 0,
                        'verbose': 0,  # 1
                        'error_on_fail': 1}

        # Create a solver that uses IPOPT with above solver settings
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.SteerSetParams2 = SteerSetParams2(dt, f, N, solver, argums, numStates, numControls)

    def solveNLP(self, solver, argums, x0, xT, n_states, n_controls, N, T): # TODO: COME BACK TO THIS LATER TO PROPAGATE COVARS
        """
        Solves the nonlinear steering problem using the solver from SetUpSteeringLawParameters
        Inputs:
            solver: Casadi NLP solver from SetUpSteeringLawParameters
            argums: argums with lbg, ubg, lbx, ubx
            x0, xT: initial and final states as (n_states)x1 ndarrays e.g. [[2.], [4.], [3.14]]
            n_states, n_controls: number of states and controls
            N: horizon
            T: time step
        Outputs:
            x_casadi, u_casadi: trajectory states and inputs returned by Casadi
                if solution found:
                    states: (N+1)x(n_states) ndarray e.g. [[1  2  0], [1.2  2.4  0], [2  3.5  0]]
                    controls: (N)x(n_controls) ndarray e.g. [[0.5  0], [1  0.01], [1.2  -0.01]]
                else, [],[] returned
        """

        # Create an initial state trajectory that roughly accomplishes the desired state transfer (by interpolating)
        init_states_param = np.linspace(0, 1, N + 1)
        init_states = np.zeros([N + 1, n_states])
        dx = xT - x0
        for i in range(N + 1):
            init_states[i] = (x0 + init_states_param[i] * dx).flatten()

        # Create an initial input trajectory that roughly accomplishes the desired state transfer
        # (using interpolated states to compute rough estimate of controls)
        dist = la.norm(xT[0:2] - x0[0:2])
        ang_dist = xT[2][0] - x0[2][0]
        total_time = N * T
        const_vel = dist / total_time
        const_ang_vel = ang_dist / total_time
        init_inputs = np.array([const_vel, const_ang_vel] * N).reshape(-1, 2)

        # Initialize and solve NLP
        c_p = np.concatenate((x0, xT))  # start and goal state constraints
        init_decision_vars = np.concatenate((init_inputs.reshape(-1, 1), init_states.reshape(-1, 1)))
        try:
            res = solver(x0=init_decision_vars, p=c_p, lbg=argums['lbg'], lbx=argums['lbx'], ubg=argums['ubg'],
                         ubx=argums['ubx'])
        except:
            # raise Exception('NLP failed')
            # print('NLP Failed')
            return [], []

        # Output series looks like [u0, u1, ..., uN, x0, x1, ..., xN+1]  #####[u0, x0, u1, x1, ...]
        casadi_result = res['x'].full()

        # Extract the control inputs and states
        u_casadi = casadi_result[:2 * N].reshape(n_controls, N).T  # (N, n_controls)
        x_casadi = casadi_result[2 * N:].reshape(n_states, N + 1).T  # (N+1, n_states)

        return x_casadi, u_casadi

    def nonlinsteer(self, steerParams): # TODO:COME BACK TO THIS LATER TO EDIT
        """
        Use created solver from `SetUpSteeringLawParameters` to solve NLP using `solveNLP` then rearrange the results
        """

        # Unbox the input parameters
        fromNode = steerParams["fromNode"]
        toNode = steerParams["toNode"]
        SteerSetParams = steerParams["Params"]

        # Unwrap the variables needed for simulation
        N = SteerSetParams.N  # Steering horizon
        solver = SteerSetParams.solver  # NLP IPOPT Solver object
        argums = SteerSetParams.argums  # NLP solver arguments
        numStates = SteerSetParams.numStates  # Number of states
        numControls = SteerSetParams.numControls  # Number of controls

        # Feed the source and destination parameter values
        x0 = fromNode.means[-1, :, :]  # source
        xGoal = toNode.means[-1, :, :]  # destination

        [x_casadi, u_casadi] = self.solveNLP(solver, argums, x0, xGoal, numStates, numControls, N, DT)

        if x_casadi == []:  # NLP problem failed to find a solution
            steerResult = False
            steerOutput = {"means": [],
                           "covars": [],
                           "cost": [],
                           "steerResult": steerResult,
                           "inputCommands": []}
            return steerOutput

        # else, steering is successful
        steerResult = True
        xHist = []
        uHist = []

        # add states/controls to xHist/uHist
        for i in range(len(u_casadi)):
            xHist.append(x_casadi[i, :].reshape(numStates, 1))
            uHist.append(u_casadi[i, :])
        xHist.append(x_casadi[-1, :].reshape(numStates, 1))

        # compute steering cost
        Q = self.Q
        R = self.R
        QT = self.Q
        steeringCost = 0
        xGoal3 = xGoal.reshape(numStates) # Goal node we tried to steer to
        for i in range(len(uHist)):
            cost_i = 0
            # # state cost
            # state_i = copy.copy(xHist[i])
            # state_i = state_i.reshape(numStates) # subtract the desired location
            # cost_i += (state_i-xGoal3).dot(Q).dot(state_i-xGoal3)
            # control effort cost
            ctrl_i = copy.copy(uHist[i])
            ctrl_i = ctrl_i.reshape(numControls)
            cost_i += ctrl_i.dot(R).dot(ctrl_i)
            # update steering cost
            steeringCost += cost_i
        # add cost on final state relative to goal. should always be zero
        state_i = copy.copy(xHist[i + 1])
        state_i = state_i.reshape(numStates)  # subtract the desired location
        cost_i = (state_i - xGoal3).dot(QT).dot(state_i - xGoal3)
        steeringCost += cost_i

        # Find covariances
        # Find covariances
        if DRRRT:
            covarHist = self.ukfCovars(xHist, uHist, steerParams)
        else:
            covarHist = [np.zeros([numStates, numStates])] * (N + 1)

        # Prepare output dictionary
        steerOutput = {"means": xHist, #TODO: ADD COVARS
                       "covars": covarHist, # TODO: CHECK THIS AND ITS DIM'NS
                       "cost": steeringCost,
                       "steerResult": steerResult,
                       "inputCommands": uHist}
        return steerOutput

    def ukfCovars(self, xHist, uHist, steerParams):

        # Unbox the input parameters
        fromNode = steerParams["fromNode"]
        SteerSetParams = steerParams["Params"]

        # Unwrap the variables needed for simulation
        N = SteerSetParams.N  # Steering horizon
        numStates = self.numStates  # Number of states

        ukf_params = {}
        ukf_params["n_x"] = self.numStates
        ukf_params["n_o"] = self.numOutputs
        ukf_params["SigmaW"] = self.SigmaW
        ukf_params["SigmaV"] = self.SigmaV
        ukf_params["CrossCor"] = self.CrossCor # TODO: DEFINED THIS IS __init__
        ukf_params["dT"] = DT

        # Find covariances
        SigmaE = fromNode.covars[-1, :, :]  # covariance at initial/from node
        covarHist = [SigmaE]
        for k in range(0, N): # TODO: is this up to N or N-1
            x_hat = xHist[k] # TODO: k-th state?
            u_k = uHist[k] # TODO: k-th control? ALSO CHECK DIM'N
            y_k = xHist[k+1] # (we assume perfect full state feedback so y = x) TODO: k+1-th measurement = k+1 state?

            ukf_params["x_hat"] = x_hat
            ukf_params["u_k"] = u_k
            ukf_params["SigmaE"] = SigmaE
            ukf_params["y_k"] = y_k

            ukf_estimator = UKF_Estimator.UKF()  # initialize the state estimator
            estimator_output = ukf_estimator.Estimate(ukf_params)  # get the estimates
            x_hat = np.squeeze(estimator_output["x_hat"]) # Unbox the state (this is the same as the input x_hat = xHist so we don't need it)
            SigmaE = estimator_output["SigmaE"] # Unbox the covariance
            covarHist.append(SigmaE.reshape(numStates, numStates))

        return covarHist

    ###########################################################################

    def PerformCollisionCheck(self, xTrajs):
        """
        Performs point-obstacle & line-obstacle check in distributionally robust fashion.
        Input Parameters:
        xTrajs: collection of means of points along the steered trajectory
        Outputs:
        Ture if safe, Flase if collision
        """
        for k, xTraj in enumerate(xTrajs):
            if k != 0:
                # DR - Point-Obstacle Collision Check
                # collisionFreeFlag = True: Safe Trajectory and False: Unsafe Trajectory
                drCollisionFreeFlag = self.DRCollisionCheck(xTrajs[k])
                if not drCollisionFreeFlag:
                    # print('Point-Obstacle Collision Detected :::::::::')
                    return False
                    # # DR - Line-Obstacle Collision Check via LTL specifications
                drSTLCollisionFreeFlag = self.DRSTLCollisionCheck(xTrajs[k - 1], xTrajs[k])
                if not drSTLCollisionFreeFlag:
                    # print('Line-Obstacle Collision Detected ---------')
                    return False
                    # If everything is fine, return True
        return True

    ###########################################################################
    def DRCollisionCheck(self, trajNode): #TODO: CHECK LATER
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Inputs:
        trajNode  : Node containing data to be checked for collision
        Outputs:
        True if safe, False if collision
        """
        # Define the direction arrays
        xDir = np.array([1, 0, 0])
        yDir = np.array([0, 1, 0])

        # Initialize the flag to be true
        drCollisionFreeFlag = True

        for alpha, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):
            # Check if the trajNode is inside the bloated obstacle (left and right and bottom and top)
            # TODO: CHECK HOW THE 2-NORM TERM IS FOUND LATER
            Delta = math.sqrt((1-alpha)/alpha)
            # print('padding: ', Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir))
            if trajNode.X[0] >= (ox - Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir)) and \
                trajNode.X[0] <= (ox + wd + Delta*math.sqrt(xDir.T @ trajNode.Sigma @ xDir) ) and \
                trajNode.X[1] <= (oy - Delta*math.sqrt(yDir.T @ trajNode.Sigma @ yDir)) and \
                trajNode.X[1] >= (oy + ht + Delta*math.sqrt(yDir.T @ trajNode.Sigma @ yDir)):
                # collision has occured, so return false
                drCollisionFreeFlag = False
                return drCollisionFreeFlag

        return drCollisionFreeFlag  # safe, so true

    ###########################################################################

    def DRSTLCollisionCheck(self, firstNode, secondNode): # TODO: CHECK LATER
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Input Parameters:
        firstNode  : 1st Node containing data to be checked for collision
        secondNode : 2nd Node containing data to be checked for collision
        """
        xDir = np.array([1, 0, 0])
        yDir = np.array([0, 1, 0])

        # Get the coordinates of the Trajectory line connecting two points
        x1 = firstNode.X[0]
        y1 = firstNode.X[1]
        x2 = secondNode.X[0]
        y2 = secondNode.X[1]

        itr = 0
        for alpha, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList):
            itr += 1
            Delta = math.sqrt((1 - alpha) / alpha)
            # Prepare bloated version of min and max x,y positions of obstacle
            minX = ox - Delta * math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
            minY = oy - Delta * math.sqrt(yDir.T @ secondNode.Sigma @ yDir)
            maxX = ox + wd + Delta * math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
            maxY = oy + ht + Delta * math.sqrt(yDir.T @ secondNode.Sigma @ yDir)


            # Condition for Line to be Completely outside the rectangle
            if (x1 <= minX and x2 <= minX or
                    y1 <= minY and y2 <= minY or
                    x1 >= maxX and x2 >= maxX or
                    y1 >= maxY and y2 >= maxY):
                continue

            # Calculate the slope of the line
            lineSlope = (y2 - y1) / (x2 - x1)

            # Connect with a line to other point and check if it lies inside
            yPoint1 = lineSlope * (minX - x1) + y1
            yPoint2 = lineSlope * (maxX - x1) + y1
            xPoint1 = (minY - y1) / lineSlope + x1
            xPoint2 = (maxY - y1) / lineSlope + x1

            if (yPoint1 > minY and yPoint1 < maxY or
                    yPoint2 > minY and yPoint2 < maxY or
                    xPoint1 > minX and xPoint1 < maxX or
                    xPoint2 > minX and xPoint2 < maxX):
                # print('COLLISION DETECTED !!!!!!!!!!!!!!!!!!!!!')
                # print('Obstacle number:', itr)
                # print('Obstacle: ', [ox, oy, wd, ht])
                # print('x1,y1:',x1,y1)
                # print('x2,y2:', x2, y2)
                # print('minX', minX)
                # print('minY', minY)
                # print('maxX', maxX)
                # print('maxY', maxY)
                return False

        return True  # Collision Free - No Interection

    ###########################################################################

    def PrepareMinNode(self, nearestIndex, xTrajs, trajCost):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters:
        nearestIndex : Index of the nearestNode in the DR-RRT* tree
        xTrajs       : Trajectory data containing the sequence of means
        trajCost     : Cost to Steer from nearestNode to randNode
        inputCommands: Input commands needed to steer from nearestNode to randNode
        """
        # Convert trajNode to DR-RRT* Tree Node
        num_traj_nodes = len(xTrajs)
        minNode = DR_RRTStar_Node(self.numStates, self.numControls, num_traj_nodes)
        # Associate the DR-RRT* node with sequence of means
        for k, xTraj in enumerate(xTrajs):
            minNode.means[k, :, :] = xTraj.X
            minNode.covars[k, :, :] = xTraj.Sigma
            if k < num_traj_nodes - 1:
                minNode.inputCommands[k, :] = xTraj.Ctrl
        minNode.cost = self.nodeList[nearestIndex].cost + trajCost
        # Associate MinNode's parent as NearestNode
        minNode.parent = nearestIndex
        return minNode

    ###########################################################################

    def FindNearNodeIndices(self, randNode):
        """
        Returns indices of all nodes that are closer to randNode within a specified radius
        Input Parameters:
        randNode : Node around which the nearest indices have to be selected
        """
        totalNodes = len(self.nodeList)
        searchRadius = ENVCONSTANT * ((math.log(totalNodes + 1) / totalNodes + 1)) ** (1 / 2)
        distanceList = []
        for node in self.nodeList:
            distanceList.append(self.ComputeL2Distance(node, randNode))
        nearIndices = [distanceList.index(dist) for dist in distanceList if dist <= searchRadius ** 2]
        return nearIndices

    ###########################################################################

    def ConnectViaMinimumCostPath(self, nearestIndex, nearIndices, randNode, minNode): # TODO: DOUBLE CHECK THAT NO OTHER CHANGES ARE REQUIRED
        """
        Chooses the minimum cost path by selecting the correct parent
        Input Parameters:
        nearestIndex : Index of DR-RRT* Node that is nearest to the randomNode
        nearIndices  : Indices of the nodes that are nearest to the randNode
        randNode     : Randomly sampled node
        minNode      : randNode with minimum cost sequence to connect as of now
        """
        # If the queried node is a root node, return the same node
        if not nearIndices:
            return minNode

        # If there are other nearby nodes, loop through them
        for j, nearIndex in enumerate(nearIndices):
            # Looping except nearestNode - Uses the overwritten equality check function
            if self.nodeList[nearIndex] == self.nodeList[nearestIndex]:
                continue
            # if nearIndex == nearestIndex: # TODO: Try this instead
            #     continue

            # Try steering from nearNode to randNodeand get the trajectory
            success_steer, temp_minNode = self.SteerAndGetMinNode(from_idx=nearIndex, to_node=randNode) # TODO: EDIT THIS FUNCTION

            # If steering failed, move on
            if not success_steer:
                continue
            # If steering succeeds, check if the new steering cost is less than the current minNode steering cost
            if temp_minNode.cost < minNode.cost:
                # If lower cost, update minNode
                minNode = copy.copy(temp_minNode)

        return minNode

    ###########################################################################

    def ReWire(self, nearIndices, minNode): # TODO: DOUBLE CHECK - EDITED
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode     : randNode with minimum cost sequence to connect as of now
        """
        # Get all ancestors of minNode
        minNodeAncestors = self.GetAncestors(minNode)
        for j, nearIndex in enumerate(nearIndices):
            # Avoid looping all ancestors of minNode
            if np.any([self.nodeList[nearIndex] == minNodeAncestor for minNodeAncestor in minNodeAncestors]):
                continue

            if len(nearIndices) > SBSPAT and npr.rand() < 1 - (SBSP / 100):
                continue

            # steer from the minNode to the nodes around it with nearIndex and find the trajectory
            success_steer, xTrajs, sequenceCost = self.SteerAndGenerateTrajAndCostWithFinalHeading(from_node=minNode, to_idx=nearIndex)

            # If steering fails, move on
            if not success_steer:
                continue

            connectCost = minNode.cost + sequenceCost
            # Proceed only if J[x_min] + del*J(sigma,pi) < J[X_near]
            if connectCost < self.nodeList[nearIndex].cost:
                self.nodeList[nearIndex].parent = len(self.nodeList) - 1
                self.nodeList[nearIndex].cost = connectCost
                # prepare the means and inputs
                num_traj_nodes = len(xTrajs)
                meanSequence = np.zeros((num_traj_nodes, self.numStates, 1))  # Mean Sequence
                covarSequence = np.zeros((num_traj_nodes, self.numStates, self.numStates))  # Covar Sequence
                inputCtrlSequence = np.zeros((num_traj_nodes - 1, self.numControls))  # Input Sequence
                for k, xTraj in enumerate(xTrajs):
                    meanSequence[k, :, :] = xTraj.X
                    covarSequence[k, :, :] = xTraj.Sigma
                    if k < num_traj_nodes - 1:
                        inputCtrlSequence[k, :] = xTraj.Ctrl
                # overwrite the mean and inputs sequences in the nearby node
                self.nodeList[nearIndex].means = meanSequence  # add the means from xTrajs
                self.nodeList[nearIndex].covars = covarSequence  # add the covariances from xTrajs
                self.nodeList[nearIndex].inputCommands = inputCtrlSequence  # add the controls from xTrajs
                # Update the children of nearNode about the change in cost
                rewire_count = 0
                self.UpdateDescendantsCost(self.nodeList[nearIndex], rewire_count)

    ###########################################################################

    def UpdateDescendantsCost(self, newNode, rewire_count):  # TODO: DOUBLE CHECK - EDITED
        """
        Updates the cost of all children nodes of newNode
        Input Parameter:
        newNode: Node whose children's costs have to be updated
        """
        rewire_count += 1
        if rewire_count > MAXDECENTREWIRE:
            return
        # Record the index of the newNode
        newNodeIndex = self.nodeList.index(newNode)
        # Loop through the nodeList to find the children of newNode
        for childNode in self.nodeList[newNodeIndex:]:
            # Ignore Root node and all ancestors of newNode - Just additional check
            if childNode.parent is None or childNode.parent < newNodeIndex:
                continue
            if childNode.parent == newNodeIndex:
                success_steer, xTrajs, trajCost = self.SteerAndGenerateTrajAndCostWithFinalHeading(from_idx=newNodeIndex,
                                                                                                   to_node=childNode)
                if not success_steer:
                    continue

                childNode.cost = newNode.cost + trajCost

                # prepare the means and inputs
                num_traj_nodes = len(xTrajs)
                meanSequence = np.zeros((num_traj_nodes, self.numStates, 1))  # Mean Sequence
                covarSequence = np.zeros((num_traj_nodes, self.numStates, self.numStates))  # Covar Sequence
                inputCtrlSequence = np.zeros((num_traj_nodes - 1, self.numControls))  # Input Sequence
                for k, xTraj in enumerate(xTrajs):
                    meanSequence[k, :, :] = xTraj.X
                    covarSequence[k, :, :] = xTraj.Sigma
                    if k < num_traj_nodes - 1:
                        inputCtrlSequence[k, :] = xTraj.Ctrl
                # overwrite the mean and inputs sequences in the nearby node
                childNode.means = meanSequence
                childNode.covars = covarSequence
                childNode.inputCommands = inputCtrlSequence
                # Get one more level deeper
                self.UpdateDescendantsCost(childNode, rewire_count)

    ###########################################################################

    def GetGoalNodeIndex(self):
        '''
        Get incides of all RRT nodes in the goal region
        Inputs :
        NONE
        Outputs:
        goalNodeIndex: list of indices of the RRT nodes (type: python list)
        '''
        # Get the indices of all nodes in the goal area
        goalIndices = []
        for node in self.nodeList:
            if (node.means[-1, 0, :] >= self.xmingoal and node.means[-1, 1, :] >= self.ymingoal and
                    node.means[-1, 0, :] <= self.xmaxgoal and node.means[-1, 0, :] <= self.ymaxgoal):
                goalIndices.append(self.nodeList.index(node))

        # Select a random node from the goal area
        goalNodeIndex = random.choice(goalIndices)

        return goalNodeIndex

    ###########################################################################

    def GenerateSamplePath(self, goalIndex):
        '''
        Generate a list of RRT nodes from the root to a node with index goalIndex
        Inputs:
        goalIndex: index of RRT node which is set as the goal
        Outputs:
        pathNodesList: list of RRT nodes from root node to goal node (type: python list (element type: DR_RRTStar_Node)) # TODO: check this
        '''
        pathNodesList = [self.nodeList[goalIndex]]

        # Loop until the root node (whose parent is None) is reached
        while self.nodeList[goalIndex].parent is not None:
            # Set the index to its parent
            goalIndex = self.nodeList[goalIndex].parent
            # Append the parent node to the pathNodeList
            pathNodesList.append(self.nodeList[goalIndex])

        # Finally append the path with root node
        pathNodesList.append(self.nodeList[0])

        return pathNodesList


    ###########################################################################

    def PlotObstacles(self): # TODO: COME BACK AND EDIT THIS
        """
        Plots the obstacles and the starting position.
        """
        plot_sampled_nodes = False
        plot_tree_node_centers = True

        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
        # ax.axis('equal')
        self.ax = ax
        ax = plot_env(ax)

        ax.scatter(self.start.means[-1, 0, :], self.start.means[-1, 1, :], s=200, c='tab:blue', marker='o', label='Start')

        # plot sampled nodes
        x_sampled = []
        y_sampled = []
        xFreePoints, yFreePoints, thetaFreePoints = self.freePoints
        for i in range(len(xFreePoints)):
            if not i in self.dropped_samples: # skip nodes that became infeasible after saturation
                x_sampled.append(xFreePoints[i])
                y_sampled.append(yFreePoints[i])
        if plot_sampled_nodes:
            # plt.plot(x_sampled, y_sampled,'o', color='red', markersize=3)
            plt.plot(x_sampled, y_sampled, 'o', color='salmon', markersize=5)

        # add crosses on sampled nodes that were added to the tree
        x_added = []
        y_added = []
        for k, node in enumerate(self.nodeList):
            x_added.append(node.means[-1, 0, :][0])
            y_added.append(node.means[-1, 1, :][0])
        if plot_tree_node_centers:
            # plt.plot(x_added, y_added, 'x', color='black', markersize=5)
            plt.plot(x_added, y_added, 'o', color='black', markersize=3)

    def DrawGraph(self, lastFlag):  # TODO: COME BACK AND EDIT THIS
        """
        Updates the Plot with uncertainty ellipse and trajectory at each time step
        Input Parameters:
        lastFlag: Flag to denote if its the last iteration
        """
        plot_covars = False
        plot_only_last_covar = True
        plot_dr_check_ellipse = True

        xValues = []
        yValues = []
        widthValues = []
        heightValues = []
        angleValues = []
        lineObjects = []

        # Plot the environment with the obstacles and the starting position
        self.PlotObstacles()
        ax = self.ax

        alpha = np.array(ALFA, float)
        delta = (1 - alpha) / alpha
        delta = delta ** (0.5)
        eps = 0.0001
        all_deltas_same = all(delta[0]-eps <= elt <= delta[0]+eps for elt in list(delta))
        if not all_deltas_same:
            # if not all risk bounds are the same, plotting the dr padding on the robot doesn't make sense
            # (different paddings for every obstacle)
            plot_dr_check_ellipse = False

        for ellipseNode in self.nodeList:
            if ellipseNode is not None and ellipseNode.parent is not None:
                ellNodeShape = ellipseNode.means.shape
                xPlotValues = []
                yPlotValues = []
                # Prepare the trajectory x and y vectors and plot them
                for k in range(ellNodeShape[0]):
                    xPlotValues.append(ellipseNode.means[k, 0, 0])
                    yPlotValues.append(ellipseNode.means[k, 1, 0])
                    # Plotting the risk bounded trajectories
                lx, = plt.plot(xPlotValues,
                               yPlotValues,
                               color='#636D97',
                               linewidth=1.0)
                lineObjects.append(lx)

                if not plot_covars and not plot_dr_check_ellipse: # do not plot covars or dr_coll check
                    alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                      ellipseNode.means[-1, 0, 0])
                    xValues.append(ellipseNode.means[-1, 0, 0])
                    yValues.append(ellipseNode.means[-1, 1, 0])
                    widthValues.append(0)
                    heightValues.append(0)
                    angleValues.append(alfa * 360)
                elif not plot_covars and plot_dr_check_ellipse: # do not plot covars but plot dr_coll check ellipse
                    # Plot only the last ellipse in the trajectory
                    alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                      ellipseNode.means[-1, 0, 0])
                    elcovar = np.asarray(ellipseNode.covars[-1, :, :])  # covariance
                    # plot node dr-check (Check this... It might not make sense) TODO: CHECK
                    xValues.append(ellipseNode.means[-1, 0, 0])
                    yValues.append(ellipseNode.means[-1, 1, 0])
                    xDir = np.array([1, 0, 0])
                    yDir = np.array([0, 1, 0])
                    Delta = delta[-1]  # use environment level of padding
                    major_ax_len = (Delta * math.sqrt(
                        xDir.T @ elcovar @ xDir)) * 2  # (.) * 2 <--  because we want width of ellipse
                    minor_ax_len = (Delta * math.sqrt(
                        yDir.T @ elcovar @ yDir)) * 2  # --> padding in right and left directions added
                    widthValues.append(major_ax_len)
                    heightValues.append(minor_ax_len)
                    angleValues.append(alfa * 360)
                elif plot_only_last_covar and not plot_dr_check_ellipse: # plot covars but only at final node
                    # Plot only the last ellipse in the trajectory
                    alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                      ellipseNode.means[-1, 0, 0])
                    elcovar = np.asarray(ellipseNode.covars[-1, :, :]) # covariance

                    # plot node covariance
                    elE, elV = LA.eig(elcovar[0:2, 0:2])
                    xValues.append(ellipseNode.means[-1, 0, 0])
                    yValues.append(ellipseNode.means[-1, 1, 0])
                    widthValues.append(math.sqrt(elE[0]))
                    heightValues.append(math.sqrt(elE[1]))
                    angleValues.append(alfa * 360)
                elif plot_only_last_covar and plot_dr_check_ellipse:  # plot ellipse representing dr-check but only at final node
                    # Plot only the last ellipse in the trajectory
                    alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                      ellipseNode.means[-1, 0, 0])
                    elcovar = np.asarray(ellipseNode.covars[-1, :, :])  # covariance
                    # plot node dr-check (Check this... It might not make sense) TODO: CHECK
                    xValues.append(ellipseNode.means[-1, 0, 0])
                    yValues.append(ellipseNode.means[-1, 1, 0])
                    xDir = np.array([1, 0, 0])
                    yDir = np.array([0, 1, 0])
                    Delta = delta[-1]  # use environment level of padding
                    major_ax_len = (Delta * math.sqrt(xDir.T @ elcovar @ xDir)) * 2  # (.) * 2 <--  because we want width of ellipse
                    minor_ax_len = (Delta * math.sqrt(yDir.T @ elcovar @ yDir)) * 2  #  --> padding in right and left directions added
                    widthValues.append(major_ax_len)
                    heightValues.append(minor_ax_len)
                    angleValues.append(alfa * 360)
                    # plot node covariance
                    elE, elV = LA.eig(elcovar[0:2, 0:2])
                    xValues.append(ellipseNode.means[-1, 0, 0])
                    yValues.append(ellipseNode.means[-1, 1, 0])
                    widthValues.append(math.sqrt(elE[0]))
                    heightValues.append(math.sqrt(elE[1]))
                    angleValues.append(alfa * 360)
                elif not plot_only_last_covar: # plot covars (plot_covars=True) at all nodes (plot_only_last_covar=False)
                    for k in range(ellNodeShape[0]):
                        # Plot only the last ellipse in the trajectory
                        alfa = math.atan2(ellipseNode.means[k, 1, 0],
                                          ellipseNode.means[k, 0, 0])
                        elcovar = np.asarray(ellipseNode.covars[k, :, :])

                        elE, elV = LA.eig(elcovar[0:2, 0:2])
                        xValues.append(ellipseNode.means[-1, 0, 0])
                        yValues.append(ellipseNode.means[-1, 1, 0])
                        widthValues.append(math.sqrt(elE[0]))
                        heightValues.append(math.sqrt(elE[1]))
                        angleValues.append(alfa * 360)

        # Plot the Safe Ellipses
        XY = np.column_stack((xValues, yValues))
        ec = EllipseCollection(widthValues,
                               heightValues,
                               angleValues,
                               units='x',
                               offsets=XY,
                               facecolors="#C59434",
                               # edgecolors="b",
                               # edgecolors="#C59434",
                               transOffset=ax.transData)
        ec.set_alpha(0.3)
        ax.add_collection(ec)
        plt.pause(1.0001)
        if SAVEDATA:
            plot_name = 'plot_tree_' + FILEVERSION + '_' + SAVETIME + '.png'
            plot_name = os.path.join(SAVEPATH, plot_name)
            plt.savefig(plot_name)
        if not lastFlag:
            ec.remove()
            for lx in lineObjects:
                lx.remove()

    ###########################################################################

    ###########################################################################

    def SteerAndGenerateTrajAndCost(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        """
        Apply steering function to navigate from a starting node in the tree to a given node
        Perform a collision check
        Return the trajectory and cost between the two nodes
        Inputs:
        from_idx : index of node in the tree to navigate from
        to_node  : node to be added (DR_RRTStar_Node)
        Outputs:
        - Steering success flag (Type: bool)
        - Prepared trajectory (xTrajs) returned by PrepareTrajectory (type: # TODO: fill this)
        - Trajectory cost (type: float # TODO: CHECK THIS)
        The three outputs can have one of two options
        - True, xTrajs, trajCost: if steering succeeds (True), a trajectory is prepared (xTrajs); its cost is trajCost
        - return False, [], 0: if steering fails (False), the other parameters are set to bad values [] and 0 # TODO: consider replacing 0 with inf
        """
        # Steer from nearestNode to the randomNode using LQG Control
        # Returns a list of node points along the trajectory and cost
        # Box the steer parameters
        if from_idx == None:  # from index not given
            from_node_chosen = from_node
        else:  # from index given
            from_node_chosen = self.nodeList[from_idx]

        if to_idx == None:  # to index not given
            to_node_chosen = to_node
        else:  # to index given
            to_node_chosen = self.nodeList[to_idx]

        steerParams = {"fromNode": from_node_chosen,
                       "toNode": to_node_chosen,
                       "Params": self.SteerSetParams}
        steerOutput = self.nonlinsteer(steerParams)

        # Unbox the steer function output
        meanValues = steerOutput["means"]
        covarValues = steerOutput["covars"]
        trajCost = steerOutput["cost"]
        steerResult = steerOutput["steerResult"]
        inputCommands = steerOutput["inputCommands"]

        # If the steering law fails, force next iteration with different random sample
        if steerResult == False:
            # print('NLP Steering Failed XXXXXXXXX')
            return False, [], 0

        # Proceed only if the steering law succeeds
        # Prepare the trajectory
        xTrajs = self.PrepareTrajectory(meanValues, covarValues, inputCommands)

        # Check for Distributionally Robust Feasibility of the whole trajectory
        collisionFreeFlag = self.PerformCollisionCheck(xTrajs)

        # If a collision was detected, stop and move on
        if not collisionFreeFlag:
            # print('DR Collision Detected @@@@@@@@@')
            return False, [], 0

        return True, xTrajs, trajCost

    def SteerAndGenerateTrajAndCostWithFinalHeading(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        """
        Same as SteerAndGenerateTrajAndCost but uses the steering params, and hence the steering law, with the heading enforced to match the set value
        """
        # Steer from nearestNode to the randomNode using LQG Control
        # Returns a list of node points along the trajectory and cost
        # Box the steer parameters
        if from_idx == None:  # from index not given
            from_node_chosen = from_node
        else:  # from index given
            from_node_chosen = self.nodeList[from_idx]

        if to_idx == None:  # to index not given
            to_node_chosen = to_node
        else:  # to index given
            to_node_chosen = self.nodeList[to_idx]

        steerParams = {"fromNode": from_node_chosen,
                       "toNode": to_node_chosen,
                       "Params": self.SteerSetParams2}
        steerOutput = self.nonlinsteer(steerParams)

        # Unbox the steer function output
        meanValues = steerOutput["means"]
        covarValues = steerOutput["covars"]
        trajCost = steerOutput["cost"]
        steerResult = steerOutput["steerResult"]
        inputCommands = steerOutput["inputCommands"]

        # If the steering law fails, force next iteration with different random sample
        if steerResult == False:
            # print('NLP Steering Failed XXXXXXXXX')
            return False, [], 0

        # Proceed only if the steering law succeeds
        # Prepare the trajectory
        xTrajs = self.PrepareTrajectory(meanValues, covarValues, inputCommands)

        # Check for Distributionally Robust Feasibility of the whole trajectory
        collisionFreeFlag = self.PerformCollisionCheck(xTrajs)

        # If a collision was detected, stop and move on
        if not collisionFreeFlag:
            # print('DR Collision Detected @@@@@@@@@')
            return False, [], 0

        return True, xTrajs, trajCost

    def SteerAndGetMinNode(self, from_idx=None, from_node=None, to_idx=None, to_node=None):
        # steer and find the trajectory and trajectory cost
        success_steer, xTrajs, trajCost = self.SteerAndGenerateTrajAndCost(from_idx=from_idx, to_node=to_node)

        # If steering failed, stop
        if not success_steer:
            return False, []

        # If steering succeeds
        # Create minNode with trajectory data & Don't add to the tree for the time being
        minNode = self.PrepareMinNode(from_idx, xTrajs, trajCost)

        return True, minNode

    ###########################################################################

    def ExpandTree(self):
        """
        Subroutine that grows DR-RRT* Tree
        """

        # Prepare And Load The Steering Law Parameters
        self.SetUpSteeringLawParameters()
        self.SetUpSteeringLawParametersWithFinalHeading()

        # Generate maxIter number of free points in search space
        t1 = time.time()
        self.GetFreeRandomPoints()
        t2 = time.time()
        print('Finished Generating Free Points !!! Only took: ', t2 - t1)

        # Iterate over the maximum allowable number of nodes
        for iter in range(self.maxIter):
            print('Iteration Number', iter, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            self.iter = iter

            # Get a random feasible point in the space as a DR-RRT* Tree node
            randNode = self.GetRandomPoint()
            # print('Trying randNode: ', randNode.means[-1,1,:], randNode.means[-1,2,:])

            # Get index of best DR-RRT* Tree node that is nearest to the random node
            nearestIndex = self.GetNearestListIndex(randNode)

            # Saturate randNode
            randNode = self.SaturateNodeWithL2(self.nodeList[nearestIndex], randNode)
            xFreePoints, yFreePoints, thetaFreePoints = self.freePoints
            xFreePoints[self.iter] = randNode.means[-1, 0, :]
            yFreePoints[self.iter] = randNode.means[-1, 1, :]
            thetaFreePoints[self.iter] = randNode.means[-1, 2, :]
            if not self.RandFreeChecks(randNode.means[-1, 0, :], randNode.means[-1, 1, :]):
                self.dropped_samples.append(iter)
                continue

            success_steer, minNode = self.SteerAndGetMinNode(from_idx=nearestIndex, to_node=randNode)

            if not success_steer:
                continue

            if RRT:
                self.nodeList.append(minNode)
            else:
                # Get all the nodes in the DR-RRT* Tree that are closer to the randomNode within a specified search radius
                nearIndices = self.FindNearNodeIndices(randNode)
                # Choose the minimum cost path to connect the random node
                minNode = self.ConnectViaMinimumCostPath(nearestIndex, nearIndices, randNode, minNode)
                # Add the minNode to the DR-RRT* Tree
                self.nodeList.append(minNode)
                # Rewire the tree with newly added minNode
                self.ReWire(nearIndices, minNode)

        return self.nodeList

    ###############################################################################


########################## FUNCTIONS CALLED BY MAIN() #########################
###############################################################################

def DefineDynamics(dynamicsSelector):
    # Unicycle Dynamics (only one supported for now)
    if dynamicsSelector == 1:
        numStates = 3  # x,y,theta
        numControls = 2  # v, omega

    return Dynamics(numStates, numControls)

###############################################################################

def DefineStartParameters(dynamicsSelector):
    start = ROBSTART
    randArea = copy.copy(RANDAREA)  # [xmin,xmax,ymin,ymax]
    goal1Area = copy.copy(GOALAREA)  # [xmin,xmax,ymin,ymax]
    maxIter = NUMSAMPLES
    plotFrequency = 1
    # Obstacle Location Format [ox,oy,wd,ht]:
    # ox, oy specifies the bottom left corner of rectangle with width: wd and height: ht
    obstacleList = copy.deepcopy(OBSTACLELIST)

    # environment rectangle bottom left and top right corners
    xmin = randArea[0]
    xmax = randArea[1]
    ymin = randArea[2]
    ymax = randArea[3]
    # thickness of env edges (doesn't matter much, anything > 0  works)
    thickness = 0.1
    # original environment area - width and height
    width = xmax - xmin
    height = ymax - ymin

    # top, bottom, right, and left rectangles for the env edges
    env_bottom = [xmin-thickness, ymin-thickness, width+2*thickness, thickness]
    env_top = [xmin-thickness, ymax, width+2*thickness, thickness]
    env_right = [xmax, ymin-thickness, thickness, height+2*thickness]
    env_left = [xmin-thickness, ymin-thickness, thickness, height+2*thickness]

    obstacleList.append(env_bottom)
    obstacleList.append(env_top)
    obstacleList.append(env_right)
    obstacleList.append(env_left)

    # TODO: add env to obstacleList. Might have to remove randArea padding

    # add padding to randArea bounds:
    randArea[0] += ROBRAD  # increase minimum x by robot radius
    randArea[1] -= ROBRAD  # decrease maximum x by robot radius
    randArea[2] += ROBRAD  # increase minimum y by robot radius
    randArea[3] -= ROBRAD  # decrease maximum y by robot radius

    # add enough padding for obstacles for robot radius
    for obs in obstacleList:
        obs[0] -= ROBRAD  # decrease bottom left corner along x direction by robot radius
        obs[1] -= ROBRAD  # decrease bottom left corner along y direction by robot radius
        obs[2] += (2 * ROBRAD)  # increase width of obstacle by robot diameter
        obs[3] += (2 * ROBRAD)  # increase height of obstacle by robot diameter

    Dynamics = DefineDynamics(dynamicsSelector)

    return StartParams(start, randArea, goal1Area, maxIter, plotFrequency, obstacleList, Dynamics)


###############################################################################
def plot_tree(ax, pathNodesList, filename):

    plot_dr_check_ellipse = True
    xValues = []
    yValues = []
    widthValues = []
    heightValues = []
    angleValues = []
    lineObjects = []

    alpha = np.array(ALFA, float)
    delta = (1 - alpha) / alpha
    delta = delta ** (0.5)
    eps = 0.0001
    all_deltas_same = all(delta[0] - eps <= elt <= delta[0] + eps for elt in list(delta))
    if not all_deltas_same:
        # if not all risk bounds are the same, plotting the dr padding on the robot doesn't make sense
        # (different paddings for every obstacle)
        plot_dr_check_ellipse = False

    for ellipseNode in pathNodesList:
        if ellipseNode is not None and ellipseNode.parent is not None:
            ellNodeShape = ellipseNode.means.shape
            xPlotValues = []
            yPlotValues = []
            # Prepare the trajectory x and y vectors and plot them
            for k in range(ellNodeShape[0]):
                xPlotValues.append(ellipseNode.means[k, 0, 0])
                yPlotValues.append(ellipseNode.means[k, 1, 0])
                # Plotting the risk bounded trajectories
            lx, = ax.plot(xPlotValues,
                           yPlotValues,
                           color='#0078f0',
                           linewidth=1.0,
                           alpha=0.9)
            lineObjects.append(lx)

            if not plot_dr_check_ellipse:  # do not plot dr_coll check
                alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                  ellipseNode.means[-1, 0, 0])
                xValues.append(ellipseNode.means[-1, 0, 0])
                yValues.append(ellipseNode.means[-1, 1, 0])
                widthValues.append(0)
                heightValues.append(0)
                angleValues.append(alfa * 360)
            elif plot_dr_check_ellipse:  # plot dr_coll check ellipse
                # Plot only the last ellipse in the trajectory
                alfa = math.atan2(ellipseNode.means[-1, 1, 0],
                                  ellipseNode.means[-1, 0, 0])
                elcovar = np.asarray(ellipseNode.covars[-1, :, :])  # covariance
                # plot node dr-check (Check this... It might not make sense) TODO: CHECK
                xValues.append(ellipseNode.means[-1, 0, 0])
                yValues.append(ellipseNode.means[-1, 1, 0])
                xDir = np.array([1, 0, 0])
                yDir = np.array([0, 1, 0])
                Delta = delta[-1]  # use environment level of padding
                major_ax_len = (Delta * math.sqrt(
                    xDir.T @ elcovar @ xDir)) * 2  # (.) * 2 <--  because we want width of ellipse
                minor_ax_len = (Delta * math.sqrt(
                    yDir.T @ elcovar @ yDir)) * 2  # --> padding in right and left directions added
                widthValues.append(major_ax_len)
                heightValues.append(minor_ax_len)
                angleValues.append(alfa * 360)

    # Plot the Safe Ellipses
    XY = np.column_stack((xValues, yValues))
    ec = EllipseCollection(widthValues,
                           heightValues,
                           angleValues,
                           units='x',
                           offsets=XY,
                           # facecolors='#dda032',
                           # facecolors='#3c3359',
                           # facecolors='#0078f0',
                           # facecolors='#f6b337',
                           # facecolors='#345160',
                           facecolors="#C59434",
                           # edgecolors="b",
                           transOffset=ax.transData,
                           alpha=0.5)
    ax.add_collection(ec)

    axis_limits = [-5.2, 5.2, -5.2, 5.2]
    ax.axis('equal')
    ax.axis(axis_limits)
    ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.autoscale(False)

    plt.pause(1.0001)
    if SAVEDATA:
        plot_filename = filename.replace('NodeListData', "plot_tree")
        # plot_name = plot_filename + '_reconstructed' + '.png'
        plot_name = plot_filename + '.png'
        plot_name = os.path.join(SAVEPATH, plot_name)
        plt.savefig(plot_name)
    return ax


def plot_saved_data(pathNodesList, filename, show_start=True):
    plot_sampled_nodes = False
    plot_tree_node_centers = True

    fig, ax = plt.subplots(figsize=[4, 4])
    ax = plot_env(ax)

    start = pathNodesList[0]
    if show_start:
        ax.scatter(start.means[-1, 0, :], start.means[-1, 1, :], s=200, c='w', edgecolor='k', linewidths=2, marker='^',
                   label='Start', zorder=200)

    # add crosses on sampled nodes that were added to the tree
    x_added = []
    y_added = []
    for k, node in enumerate(pathNodesList):
        x_added.append(node.means[-1, 0, :][0])
        y_added.append(node.means[-1, 1, :][0])
    if plot_tree_node_centers:
        plt.plot(x_added, y_added, 'o', color='black', markersize=3)

    ax = plot_tree(ax, pathNodesList, filename)
    plt.show()
    return fig, ax


def load_and_plot(filename):
    filename1 = os.path.join(SAVEPATH, filename)
    with open(filename1, 'rb') as f:
        pathNodesList = pickle.load(f)

    plot_saved_data(pathNodesList, filename)


###############################################################################
########################## MAIN() FUNCTION ####################################
###############################################################################

def main():
    ######################## Select Desired Dynamics###########################
    # Define the Dynamics Selector Flag - 1:Unicycle, 2:Bicycle, 3:Car-Robot, 4:Quadrotor
    dynamicsSelector = 1  # only unicycle is currently supported

    # Define the Starting parameters
    startParam = DefineStartParameters(dynamicsSelector)
    start, randArea, goal1Area, maxIter, plotFrequency, obstacleList, Dynamics = startParam

    ######################## Grow DR-RRTStar Tree #############################
    # Start the timer
    t1 = time.time()

    # Create the DR_RRTStar Class Object by initizalizng the required data
    dr_rrtstar = DR_RRTStar(startParam)

    # Perform DR_RRTStar Tree Expansion
    pathNodesList = dr_rrtstar.ExpandTree()

    # Stop the timer
    t2 = time.time()
    ################### Done Growing the DR-RRTStar Tree ########################

    print("Number of Nodes:", len(pathNodesList))
    # Print Final Tree Information
    fmt = '{:<8}{:<20}{}'
    print("Printing Final Tree Information")
    print(fmt.format('Node_ID', 'x', 'y'))
    for k, node in enumerate(pathNodesList):
        print(fmt.format(k,
                         round(node.means[-1, 0, :][0], 2),
                         round(node.means[-1, 1, :][0], 2)))

    # display the total time & number of nodes
    print('Elapsed Total Time:', t2 - t1, ' seconds')
    print('Time suffix for saved files: ', SAVETIME)

    ######################## Plot DR-RRTStar Tree #############################

    # plot the Tree
    dr_rrtstar.DrawGraph(1)

    ######################## Save DR-RRTStar Tree Data ########################

    # Pickle the nodeList data and dump it for further analysis and plot
    if SAVEDATA:
        filename = 'NodeListData_' + FILEVERSION + '_' + SAVETIME
        filename = os.path.join(SAVEPATH, filename)
        outfile = open(filename, 'wb')
        pickle.dump(pathNodesList, outfile)
        outfile.close()

def main_from_data():
    # filename = 'NodeListData_v1_0_1614543353'
    # filename = 'NodeListData_v1_0_1614552959'
    filename = 'NodeListData_v1_0_1625163893'
    load_and_plot(filename)

###############################################################################

if __name__ == '__main__':
    # Close any existing figure
    plt.close('all')
    run_rrt = True
    if run_rrt:
        main()
    else:
        main_from_data()

###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################
