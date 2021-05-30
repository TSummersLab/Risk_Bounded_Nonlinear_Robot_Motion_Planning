# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:10:03 2020

@author: vxr131730 - Venkatraman Renganathan
This script simulates Path Planning with Distributionally Robust Constrained RRT*
A nonlinear model predictive control based steering law is used along with the
state estimation performed using Unscented Kalman Filter having cross correlated
process and sensor noises. Risk quantification is performed using Signal Temporal
 Logic Based Formulation. 
This script is tested in Python 3.7, Windows 10, 64-bit
(C) Venkatraman Renganathan, 2021.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3.7, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""

###############################################################################
###############################################################################

# Import all the required libraries
import random
import time
import math
import numpy as np
import pickle
from numpy import linalg as LA
from namedlist import namedlist
from scipy.special import erfinv

###############################################################################
###############################################################################

# Defining Global Variables
ENVCONSTANT = 30  # Environment Constant for computing search radius # Should be 30

# Define the namedlists
Dynamics = namedlist("Dynamics", "numStates numControls numOutputs S0 SigmaW SigmaV")
StartParams = namedlist("StartParams", "start randArea goalArea maxIter obstacleList env_obstacles dynamicsData")
SteerSetParams = namedlist("SteerSetParams", "dt f N solver ukfParam numStates numControls")

###############################################################################
###############################################################################

class trajNode():
    """
    Class Representing a steering law trajectory Node
    """ 
    
    def __init__(self, numStates):
        """
        Constructor Function
        """
        self.X     = np.zeros((numStates, 1))         # State Vector 
        self.Sigma = np.zeros((numStates, numStates)) # Covariance Matrix

###############################################################################
###############################################################################
        
class DR_RRTStar_Node():
    """
    Class Representing a DR_RRT* Tree Node
    """
    
    def __init__(self, numStates, numNodes):
        """
        Constructor Function
        """                  
        self.cost   = 0.0                                        # Cost         
        self.parent = None                                       # Index of the parent node       
        self.means  = np.zeros((numNodes, numStates, 1))         # Mean Sequence
        self.covar  = np.zeros((numNodes, numStates, numStates)) # Covariance Sequence        
    
    ###########################################################################
    
    def __eq__(self, other):
        """
        Overwriting equality check function to compare two same class objects
        """
        costFlag   = self.cost == other.cost
        parentFlag = self.parent == other.parent
        meansFlag  = np.array_equal(self.means, other.means)
        covarFlag  = np.array_equal(self.covar, other.covar)
        
        return costFlag and parentFlag and meansFlag and covarFlag       
        
###############################################################################
###############################################################################
class DR_RRTStar():
    """
    Class for DR_RRT* Planning
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
        start, randArea, goalArea, maxIter, obstacleList, env_obstacles, Dynamics = startParam
                
        # Add the Double Integrator Data  
        self.iter           = 0
        self.controlPenalty = 0.02       
        self.xminrand       = randArea[0]
        self.xmaxrand       = randArea[1]               
        self.yminrand       = randArea[2]
        self.ymaxrand       = randArea[3]  
        self.xmingoal       = goalArea[0]
        self.xmaxgoal       = goalArea[1]
        self.ymingoal       = goalArea[2]
        self.ymaxgoal       = goalArea[3]             
        self.maxIter        = maxIter                
        self.obstacleList   = obstacleList 
        self.env_obstacles  = env_obstacles 
        self.dynamicsData   = Dynamics        
        self.alfaThreshold  = 0.05
        self.alfa           = [self.alfaThreshold]*len(self.obstacleList)
        self.goalOrient     = start[2] # Same orientation as the starting point of the car
        self.obstacle_x     = start[4]
        self.obstacle_y     = start[5]
        self.egoCarLength   = start[6]
        self.egoCarWidth    = start[7]
        self.desDist        = start[8]

        # Initialize DR-RRT* tree node with start coordinates                 
        self.InitializeTree(start)        
         
    ###########################################################################
    
    def InitializeTree(self, start):
        """
        Prepares DR-RRT* tree node with start coordinates & adds to nodeList
        """     
        
        # Unwrap the dynamicsData
        self.numStates, numControls, numOutputs, self.S0, SigmaW, SigmaV = self.dynamicsData
        
        # Create an instance of DR_RRTStar_Node class
        numNodes = 1
        self.start = DR_RRTStar_Node(self.numStates, numNodes)                                 
        
        # Set the covariance sequence to the initial condition value
        for k in range(numNodes):
            self.start.covar[k,:,:] = self.S0  
            for i in range(self.numStates):
                self.start.means[k,i,:] = start[i]        
        
        # Add the created start node to the nodeList
        self.nodeList = [self.start]          
        
    ###########################################################################
    ###########################################################################
    
    def ComputeLyapunovDistance(self, fromNode, toNode):
        """
        Returns the distance between two nodes computed using the dynamic control-based distance metric
        Input parameters:
        fromNode   : Node representing point A
        toNode     : Node representing point B        
        """
        
        # Use the dynamic control-based distance metric
        diffVec = (fromNode.means[-1,:,:] - toNode.means[-1,:,:])[0:2,0]
        
        # Define the scaling constants for the robot
        kPhi = 1.2
        kDel = 3  
        
        # Compute the radial (Euclidean) target distance
        r = LA.norm(diffVec)        
        
        # Orientation of the vehicle heading w.r.t LOS - Called Del in (-pi, pi]
        Del = -2*np.pi + math.atan2(toNode.means[-1,1,:], toNode.means[-1,0,:])
        
        # Orientation of target pose w.r.t. LOS - Called Phi in (-pi, pi]
        # toNode.means[2] is in degrees - convert to radians and add delta
        Phi = toNode.means[-1,2,:]*(np.pi/180) + Del
        
        # desired heading points directly to target pose
        dDel = 0   
        
        # Compute the distance
        dist = math.sqrt(r**2 + (kPhi*Phi)**2) + kDel*abs(Del - dDel)

        return dist              
    
    ###########################################################################
    
    def RandFreeChecks(self, x, y):
        """
        Performs Collision Check For Random Sampled Point
        Input Parameters:
        x,y : Position data which has to be checked for collision 
        """
        # Get car clearance
        carTolerance = self.egoCarLength # math.sqrt((self.egoCarLength/2)**2 + (self.egoCarWidth/2)**2)
        for ox, oy, wd, ht in self.obstacleList:                    
            if (x >= ox - carTolerance and 
                x <= ox + wd + carTolerance and 
                y >= oy + carTolerance and 
                y <= oy + ht - carTolerance):
                return False    # collision
        
        return True  # safe 
    
    ###########################################################################

    def GetRandomPoint(self):
        """
        Returns a randomly sampled node from the obstacle free space
        """ 
        # Create a randNode as a tree object
        numNodes = 1
        randNode = DR_RRTStar_Node(self.numStates, numNodes) 
        
        # Initialize using the generated free points
        xFreePoints, yFreePoints = self.freePoints
        randNode.means[-1,0,:] = xFreePoints[self.iter]
        randNode.means[-1,1,:] = yFreePoints[self.iter]
        randNode.means[-1,2,:] = random.uniform(self.goalOrient-30, self.goalOrient+30) # self.goalOrient = 270 btw (240 and 300) self.goalOrient 
        randNode.means[-1,4,:] = self.obstacle_x
        randNode.means[-1,5,:] = self.obstacle_y        
           
        return randNode  
    
    ###########################################################################
    
    def GetFreeRandomPoints(self):
        """
        Returns list of randomly sampled node points from obstacle free space
        """        
        # Store sampled points
        xFreePoints = []
        yFreePoints = []   
        
        # Flag to determine if random or deterministic sampled points
        if self.deterministicFlag:
            # Generate deterministic sampled points            
            xFreePoints = [-75, -80, -75, -80, -75, -80, -75, -80, -75, -80]
            yFreePoints = [-49, -53, -57, -60, -63, -49, -53, -57, -60, -63]                        
            
        else:
            # Generate random sampled points
            for iter in range(self.maxIter): 
                # Sample uniformly around sample space                       
                if iter%50 == 0:
                    while True:
                        # initialize with a random position in space with orientation = 0
                        xPt = random.uniform(self.xmingoal, self.xmaxgoal)
                        yPt = random.uniform(self.ymingoal, self.ymaxgoal)               
                        if self.RandFreeChecks(xPt, yPt):
                            break
                else:
                    while True:
                        # initialize with a random position in space 
                        if iter < 0.14*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-5, self.ymaxrand)               
                        if 0.14*self.maxIter <= iter <= 0.28*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-10, self.ymaxrand-5)               
                        if 0.28*self.maxIter < iter <= 0.42*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-15, self.ymaxrand-10)                                       
                        if 0.42*self.maxIter <= iter <= 0.56*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-20, self.ymaxrand-15)               
                        if 0.56*self.maxIter < iter <= 0.70*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-25, self.ymaxrand-20)               
                        if 0.70*self.maxIter < iter <= 0.84*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.ymaxrand-30, self.ymaxrand-25)               
                        if iter > 0.84*self.maxIter:
                            xPt = random.uniform(self.xminrand, self.xmaxrand)
                            yPt = random.uniform(self.yminrand, self.ymaxrand-30)               
                        if self.RandFreeChecks(xPt, yPt):
                            break
                xFreePoints.append(xPt)
                yFreePoints.append(yPt)                               
                   
        self.freePoints = [xFreePoints, yFreePoints]
        
        # Save freePoints to pickle 
        filename = 'ekf_freePoints.pkl'
        outfile  = open(filename,'wb')
        pickle.dump(self.freePoints,outfile)
        outfile.close()  
        
    ###########################################################################
        
    def SaturateNodeWithDesiredDistance(self, fromNode, toNode):
        '''
        If the L2 norm of ||toNode-fromNode|| is greater than some saturation distance, 
        find a new node newToNode along the vector (toNode-fromNode) such that 
        ||newToNode-fromNode|| = saturation distance
        Inputs:
        fromNode: from/source node (type: DR_RRTStar_Node)
        toNode: desired destination node (type: DR_RRTStar_Node)
        Oupputs:
        newToNode: either the original toNode or a new one (type: DR_RRTStar_Node)
        '''
        
        # Calculate the L2 norm between fromNode and toNode
        diffVec = (fromNode.means[-1,:,:] - toNode.means[-1,:,:])[0:2,0] 
        actualDistance = LA.norm(diffVec) 
        
        if actualDistance > self.desDist:  # node is further than the saturation distance
            from_x = fromNode.means[-1, :, :][0, 0]
            from_y = fromNode.means[-1, :, :][1, 0]
            to_x = toNode.means[-1, :, :][0, 0]
            to_y = toNode.means[-1, :, :][1, 0]
            angle = math.atan2(to_y - from_y, to_x - from_x)
            new_to_x = from_x + self.desDist * math.cos(angle)
            new_to_y = from_y + self.desDist * math.sin(angle)
            newToNode = toNode
            newToNode.means[-1, :, :][0, 0] = new_to_x
            newToNode.means[-1, :, :][1, 0] = new_to_y
            return newToNode
        else:
            return toNode
        
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
            # distanceList.append(self.ComputeLyapunovDistance(node, randNode)) 
            # Make sure randNode is always above tree node
            # Reason: We dont want to steer down (taking reverse) to reach 
            # randNode from tree node            
            rand_y = randNode.means[-1, :, :][1, 0]
            tree_y = node.means[-1, :, :][1, 0]
            if rand_y < tree_y:
                distanceList.append(self.ComputeLyapunovDistance(node, randNode))                        
        return distanceList.index(min(distanceList))
    
    ###########################################################################
    
    def PrepareTrajectory(self, meanValues, covarValues):
        """
        Prepares the trajectory as trajNode from steer function outputs
        
        Input Parameters:
        meanValues : List of mean values
.        covarValues: List of covariance values
        
        Output Parameters:
        xTrajs: List of TrajNodes    
        """
        # Trajectory data as trajNode object for each steer time step
        numNodes = len(meanValues)
        xTrajs = [trajNode(self.numStates) for i in range(numNodes)] 
        for k, xTraj in enumerate(xTrajs):            
            xTraj.X     = meanValues[k].reshape(-1,1)
            xTraj.Sigma = covarValues[k] 
        
        return xTrajs

    ###########################################################################
    
    def PerformCollisionCheck(self, xTrajs):
        """
        Performs point-obstacle & line-obstacle check in distributionally robust fashion.
        Input Parameters: 
        xTrajs - collection of means & sigmas of points along the steered trajectory
        """        
        for k, xTraj in enumerate(xTrajs): 
            if k != 0:                
                # DR - Point-Obstacle Collision Check
                # collisionFreeFlag = True: Safe Trajectory and False: Unsafe Trajectory 
                drCollisionFreeFlag = self.DRCollisionCheck(xTrajs[k])
                if not drCollisionFreeFlag:
                    return False 
                # DR - Line-Obstacle Collision Check via LTL specifications
                drSTLCollisionFreeFlag = self.DRSTLCollisionCheck(xTrajs[k-1], xTrajs[k])
                if not drSTLCollisionFreeFlag:
                    return False                
        # If everything is fine, return True
        return True
    
    ###########################################################################
    def DRCollisionCheck(self, trajNode):
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance 
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Input Parameters:
        trajNode  : Node containing data to be checked for collision                 
        """               
        
        # TODO: bloat ego vehicle using its length and width
        # Define the direction arrays
        xDir = np.array([1,0,0,0,0,0])
        yDir = np.array([0,1,0,0,0,0]) 
        
        # Initialize the flag to be true 
        drCollisionFreeFlag = True
        
        # Deterministic tolerance = car length
        carTolerance = self.egoCarLength # math.sqrt((self.egoCarLength/2)**2 + (self.egoCarWidth/2)**2)
        
        leftLHS = trajNode.X[0]
        rightLHS = trajNode.X[0]
        topLHS = trajNode.X[1]
        bottomLHS = trajNode.X[1]
        
        if int(trajNode.X[0]) == 0 or int(trajNode.X[1]) == 0:
            return False
        
        # Loop through all obstacles
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList): 
            if self.DRFlag:
                # DR chance constraints
                xTol = math.sqrt((1-alfa)/alfa)*math.sqrt(xDir.T @ trajNode.Sigma @ xDir)
                yTol = math.sqrt((1-alfa)/alfa)*math.sqrt(yDir.T @ trajNode.Sigma @ yDir)
            else:
                # Gaussian chance constraints
                xTol = math.sqrt(2*xDir.T @ trajNode.Sigma @ xDir)*erfinv(1-2*alfa)
                yTol = math.sqrt(2*yDir.T @ trajNode.Sigma @ yDir)*erfinv(1-2*alfa)             
            # Check if the trajNode is inside the bloated obstacle
            if (leftLHS >= ox - carTolerance - xTol      and # Left 
                rightLHS <= ox + wd + carTolerance + xTol and # Right
                bottomLHS <= oy + carTolerance + yTol      and # Bottom
                topLHS >= oy + ht - carTolerance - yTol):   # Top
                # collision has occured, so return false             
                drCollisionFreeFlag = False
                return drCollisionFreeFlag    
            
        return drCollisionFreeFlag  # safe, so true   
    
    ###########################################################################    
    
    def LineRectangleCollisionCheck(self, ptList, extremeList, envFlag):
        
        x1,x2,y1,y2 = ptList
        minX, minY, maxX, maxY = extremeList   
        
        if envFlag:
            # Condition for Line to be Completely inside the rectangle
            if (x1 >= minX or x2 >= minX and
                y1 >= minY or y2 >= minY and
                x1 <= maxX or x2 <= maxX and
                y1 <= maxY or y2 <= maxY):
                return True
        else:
            # Condition for Line to be Completely outside the rectangle
            if (x1 <= minX and x2 <= minX or
                y1 <= minY and y2 <= minY or
                x1 >= maxX and x2 >= maxX or
                y1 >= maxY and y2 >= maxY):
                return True
    
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
            return False     

    ###########################################################################    
    
    def DRSTLCollisionCheck(self, firstNode, secondNode):
        """
        Performs Collision Check Using Deterministic Tightening of DR Chance 
        Constraint and enforces constraints to be satisfied in two successive
        time steps to avoid jumping over obstacles between the waypoints.
        Input Parameters: 
        firstNode  : 1st Node containing data to be checked for collision         
        secondNode : 2nd Node containing data to be checked for collision         
        """  
        # Set default value for collision flag
        collisionFreeFlag = True
             
        xDir = np.array([1,0,0,0,0,0])
        yDir = np.array([0,1,0,0,0,0])  
        
        # Deterministic tolerance = car length
        carTolerance = self.egoCarLength # math.sqrt((self.egoCarLength/2)**2 + (self.egoCarWidth/2)**2)        
        
        # Get the coordinates of the Trajectory line connecting two points
        x1 = firstNode.X[0]
        y1 = firstNode.X[1]
        x2 = secondNode.X[0]
        y2 = secondNode.X[1]
        ptList = [x1,x2,y1,y2]
        
        if int(x2) == 0 or int(y2) == 0:
            return False
        
        # Loop through all obstacles
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.obstacleList): 

            if self.DRFlag:
                # DR chance constraints
                xTol = math.sqrt((1-alfa)/alfa)*math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
                yTol = math.sqrt((1-alfa)/alfa)*math.sqrt(yDir.T @ secondNode.Sigma @ yDir)
            else:
                # Gaussian chance constraints
                xTol = erfinv(1-2*alfa)*math.sqrt(2*xDir.T @ secondNode.Sigma @ xDir)
                yTol = erfinv(1-2*alfa)*math.sqrt(2*yDir.T @ secondNode.Sigma @ yDir)

            # - - + +
            # Prepare bloated version of min and max x,y positions of obstacle
            minX = ox - carTolerance - xTol
            minY = oy + carTolerance + yTol
            maxX = ox + wd + carTolerance + xTol 
            maxY = oy + ht - carTolerance - yTol  
            extremeList = [minX, minY, maxX, maxY]

            # Check for collision with obstacle            
            collisionFreeFlag = self.LineRectangleCollisionCheck(ptList, extremeList, envFlag=False)
            
            if collisionFreeFlag:
                continue
            else:
                return False
            
        for alfa, (ox, oy, wd, ht) in zip(self.alfa, self.env_obstacles):
            
            if self.DRFlag:
                # DR chance constraints
                xTol = math.sqrt((1-alfa)/alfa)*math.sqrt(xDir.T @ secondNode.Sigma @ xDir)
                yTol = math.sqrt((1-alfa)/alfa)*math.sqrt(yDir.T @ secondNode.Sigma @ yDir)
            else:
                # Gaussian chance constraints
                xTol = erfinv(1-2*alfa)*math.sqrt(2*xDir.T @ secondNode.Sigma @ xDir)
                yTol = erfinv(1-2*alfa)*math.sqrt(2*yDir.T @ secondNode.Sigma @ yDir)
            
            # Prepare bloated version of min and max x,y positions of obstacle
            minX = ox - xTol
            minY = oy + yTol
            maxX = ox + wd + xTol 
            maxY = oy + ht - yTol            
            extremeList = [minX, minY, maxX, maxY]

            # Check for collision with obstacle            
            collisionFreeFlag = self.LineRectangleCollisionCheck(ptList, extremeList, envFlag=True)
            
            if collisionFreeFlag:
                continue
            else:
                return False
            
        return collisionFreeFlag # Collision Free - No Interection 

    ###########################################################################
    
    def PrepareMinNode(self, nearestIndex, xTrajs, trajCost):
        """
        Prepares and returns the randNode to be added to the DR-RRT* tree
        Input Parameters: 
        nearestIndex : Index of the nearestNode in the DR-RRT* tree        
        xTrajs       : Trajectory data containing the sequence of means and covariances
        trajCost     : Cost to Steer from nearestNode to randNode
        """
        # Convert trajNode to DR-RRT* Tree Node  
        numNodes = len(xTrajs) 
        minNode  = DR_RRTStar_Node(self.numStates, numNodes)             
        # Associate the DR-RRT* node with sequence of means and covariances data            
        for k, xTraj in enumerate(xTrajs):                        
            minNode.means[k,:,:] = xTraj.X.reshape(-1,1)                
            minNode.covar[k,:,:] = xTraj.Sigma 
        # Find mincost = Cost(x_nearest) + Line(x_nearest, x_rand)          
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
        d = 2 # dimension        
        searchRadius = ENVCONSTANT*((math.log(totalNodes) / totalNodes))**(1/(d+1)) 
        distanceList = []
        for node in self.nodeList: 
            distanceList.append(self.ComputeLyapunovDistance(node, randNode))              
        nearIndices = [distanceList.index(dist) for dist in distanceList if dist <= searchRadius**2]        
        return nearIndices
    
    ###########################################################################
    
    def ConnectViaMinimumCostPath(self, nearestIndex, nearIndices, randNode, minNode):
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
                           
        for j, nearIndex in enumerate(nearIndices):    
            # Steer only if randNode is above treeNode
            rand_y = randNode.means[-1, :, :][1, 0]
            tree_y = self.nodeList[nearIndex].means[-1, :, :][1, 0]
            # If treenode is above randnode, dont steer, just foce next iteration
            if rand_y > tree_y:
                continue
            
            # Looping except nearestNode - Uses the overwritten equality check function            
            if self.nodeList[nearIndex] == self.nodeList[nearestIndex]:                
                continue  
                
            # Try steering from nearNode to randNode and get the trajectory            
            steerOutput = self.steerObject.NMPC_Steer(self.nodeList[nearIndex], randNode)
            # Unbox the steer function output
            meanValues   = steerOutput["means"]
            covarValues  = steerOutput["covar"]
            sequenceCost = steerOutput["cost"]
            steer_status = steerOutput["status"]

            # Break the loop to force next iteration if steering failed
            if steer_status == False:
                continue                         
            
            # Prepare the trajectory
            xTrajs = self.PrepareTrajectory(meanValues, covarValues) 
            numNodes = len(xTrajs)
            xTrajsLast = xTrajs[-1]
            if int(xTrajsLast.X[0]) == 0 or int(xTrajsLast.X[1]) == 0:
                continue  
            
            # Create holders for mean and covariance sequences
            meanSequences  = np.zeros((numNodes, self.numStates, 1))
            covarSequences = np.zeros((numNodes, self.numStates, self.numStates))                
                         
            # Compute the connecting cost
            connectCost = self.nodeList[nearIndex].cost + sequenceCost 

            # Initialize collision check flags
            drCollisionFreeFlag    = True
            drSTLCollisionFreeFlag = True             
            
            # Now check for collision along the trajectory            
            for k, xTraj in enumerate(xTrajs):    
                # Update the meanSequences and covarSequences
                meanSequences[k,:,:]  = xTraj.X
                covarSequences[k,:,:] = xTraj.Sigma                
                # Check for DRSTL Feasibility                                                         
                if k != 0:
                    drCollisionFreeFlag = self.DRCollisionCheck(xTrajs[k])
                    if not drCollisionFreeFlag:
                        break 
                    # DR - Line-Obstacle Collision Check via LTL specifications
                    drSTLCollisionFreeFlag = self.DRSTLCollisionCheck(xTrajs[k-1], xTrajs[k])
                    if not drSTLCollisionFreeFlag:
                        break                                    
            # Proceed only if there is no collision
            if drSTLCollisionFreeFlag:                                                
                if connectCost < minNode.cost:   
                    # Associate minCost to connect Node as the parent of minNode
                    minNode.cost   = connectCost
                    minNode.means  = meanSequences
                    minNode.covar  = covarSequences
                    minNode.parent = nearIndex
        return minNode
    
    ###########################################################################
    
    def ReWire(self, nearIndices, minNode):
        """
        Rewires the DR-RRT* Tree using Minimum cost path found
        Input Parameters:        
        nearIndices : Indices of the nodes that are nearest to the randomNode
        minNode     : randNode with minimum cost sequence to connect as of now
        """               
            
        # Get all ancestors of minNode
        minNodeAncestors = self.GetAncestors(minNode)
        for j, nearIndex in enumerate(nearIndices):    
            
            # Steer only if minNode is below treeNode
            rand_y = minNode.means[-1, :, :][1, 0]
            tree_y = self.nodeList[nearIndex].means[-1, :, :][1, 0]
            if rand_y < tree_y:
                continue    
                              
            # Avoid looping all ancestors of minNode            
            if np.any([self.nodeList[nearIndex] == minNodeAncestor for minNodeAncestor in minNodeAncestors]):
                continue    
            
            # Steer from minNode to nearNode            
            steerOutput = self.steerObject.NMPC_Steer(minNode, self.nodeList[nearIndex])
            # Unbox the steer function output
            meanValues   = steerOutput["means"]
            covarValues  = steerOutput["covar"]
            sequenceCost = steerOutput["cost"]  
            steer_status = steerOutput["status"]

            # Break the loop to force next iteration if steering failed
            if steer_status == False:
                continue 
            
            # Prepare the trajectory
            xTrajs = self.PrepareTrajectory(meanValues, covarValues) 
            numNodes = len(xTrajs)
            xTrajsLast = xTrajs[-1]
            if int(xTrajsLast.X[0]) == 0 or int(xTrajsLast.X[1]) == 0:
                continue            
            
            # Create holders for mean and covariance sequences
            meanSequences  = np.zeros((numNodes, self.numStates, 1))
            covarSequences = np.zeros((numNodes, self.numStates, self.numStates))  
            
            # Get the connection cost
            connectCost = minNode.cost + sequenceCost
            
            # Initialize collision check flags
            drCollisionFreeFlag    = True
            drSTLCollisionFreeFlag = True     
            
            # Perform Collision Check            
            for k, xTraj in enumerate(xTrajs):             
                # Update the meanSequences and covarSequences
                meanSequences[k,:,:]  = xTraj.X
                covarSequences[k,:,:] = xTraj.Sigma
                # Check for DRSTL Feasibility                                                         
                if k != 0:
                    drCollisionFreeFlag = self.DRCollisionCheck(xTrajs[k])
                    if not drCollisionFreeFlag:
                        break 
                    # DR - Line-Obstacle Collision Check via LTL specifications
                    drSTLCollisionFreeFlag = self.DRSTLCollisionCheck(xTrajs[k-1], xTrajs[k])
                    if not drSTLCollisionFreeFlag:
                        break             
            if drSTLCollisionFreeFlag:
                # Proceed only if J[x_min] + del*J(sigma,pi) < J[X_near]                
                if connectCost < self.nodeList[nearIndex].cost:                                   
                    self.nodeList[nearIndex].parent = len(self.nodeList)-1
                    self.nodeList[nearIndex].cost   = connectCost
                    self.nodeList[nearIndex].means  = meanSequences
                    self.nodeList[nearIndex].covar  = covarSequences
                    # Update the children of nearNode about the change in cost
                    self.UpdateDescendantsCost(self.nodeList[nearIndex])                     
                        
    ###########################################################################
    
    def UpdateDescendantsCost(self, newNode):
        """
        Updates the cost of all children nodes of newNode
        Input Parameter:
        newNode: Node whose children's costs have to be updated
        """
        # Record the index of the newNode
        newNodeIndex = self.nodeList.index(newNode)
        # Loop through the nodeList to find the children of newNode
        for childNode in self.nodeList[newNodeIndex:]:            
            # Ignore Root node and all ancestors of newNode - Just additional check
            if childNode.parent is None or childNode.parent < newNodeIndex:
                continue    
            if childNode.parent == newNodeIndex:  
                # Update the sequence cost by steering from parent to child                                
                steerOutput = self.steerObject.NMPC_Steer(self.nodeList[newNodeIndex], childNode)                
                # Unbox the steer function output
                meanValues  = steerOutput["means"]
                covarValues = steerOutput["covar"]
                trajCost    = steerOutput["cost"]                
                steer_status = steerOutput["status"]

                # Break the loop to force next iteration if steering failed
                if steer_status == False:
                    continue                
                
                # Prepare the trajectory
                xTrajs = self.PrepareTrajectory(meanValues, covarValues) 
                numNodes = len(xTrajs)
                xTrajsLast = xTrajs[-1] 
                if int(xTrajsLast.X[0]) == 0 or int(xTrajsLast.X[1]) == 0:
                    continue  
                
                # Update the childNode cost             
                childNode.cost = newNode.cost + trajCost
                
                # Update the covariances and means also
                meanSequence  = np.zeros((numNodes, self.numStates, 1))
                covarSequence = np.zeros((numNodes, self.numStates, self.numStates))                
                for k, xTraj in enumerate(xTrajs):                                      
                    meanSequence[k,:,:]  = xTraj.X
                    covarSequence[k,:,:] = xTraj.Sigma
                childNode.means = meanSequence
                childNode.covar = covarSequence
                
                # Get one more level deeper
                self.UpdateDescendantsCost(childNode)    
    
    ###########################################################################
            
    def ExpandTree(self, steerLaw):        
        """
        Subroutine that grows DR-RRT* Tree using the steerLaw specified 
        """                                 
        
        # Prepare And Load The Steering Law Parameters                
        self.steerObject = steerLaw.controller.nmpc_controller
        
        # Flag to determine if random or deterministic sampled points
        # deterministicFlag = True ==> Deterministic samples (use only for debugging)
        # deterministicFlag = False ==> Random samples (actual use)
        self.deterministicFlag = False
        
        # Flag to select distributional robustness or Gaussian chance constraints
        # DRFlag = True ==> distributional robustness risk constraints
        # DRFlag = False ==> Gaussian risk constraints
        self.DRFlag = True
        
        # Generate maxIter number of free points in search space
        self.GetFreeRandomPoints()
        
        # Counter to count number of nodes in goal region
        goalNodeCounter = 0
        
        # Iterate over the maximum allowable number of nodes
        for iter in range(self.maxIter): 
            
            if iter%1 == 0:
                print('Iteration Number', iter+1)
            self.iter = iter               
                
            # Get a random feasible point in the space as a DR-RRT* Tree node
            randNode = self.GetRandomPoint()            
            
            # Get index of DR-RRT* Tree node that is nearest to random node
            nearestIndex = self.GetNearestListIndex(randNode)
            
            # If needed truncate the distance between the sampled randNode and nearestTreeNode
            # randNode = self.SaturateNodeWithDesiredDistance(self.nodeList[nearestIndex], randNode)
                
            # Check if the truncated waypoint lies in the free space. If not force next iteration
            if not self.RandFreeChecks(randNode.means[-1,0,:], randNode.means[-1,1,:]):
                continue                
            
            # Steer from nearestNode to the randomNode using NMPC
            steerOutput = self.steerObject.NMPC_Steer(self.nodeList[nearestIndex], randNode)
            
            # Unbox the steer function output
            meanValues   = steerOutput["means"]
            covarValues  = steerOutput["covar"]
            trajCost     = steerOutput["cost"]      
            steer_status = steerOutput["status"]    
            
            # Break the loop to force next iteration if steering failed
            if steer_status == False:
                # Steering Failed
                continue
            
            # Prepare the trajectory
            xTrajs = self.PrepareTrajectory(meanValues, covarValues) 

            # Get last node traj sequence
            xTrajLast = xTrajs[-1]             
            
            # Check for Distributionally Robust Feasibility of the whole trajectory              
            collisionFreeFlag = self.PerformCollisionCheck(xTrajs)
                                
            # Entire distribution sequence was DR Feasible              
            if collisionFreeFlag and int(xTrajLast.X[0,0]) != 0 and int(xTrajLast.X[1,0]) != 0:                     
                # Create minNode with trajectory data & Don't add to the tree for the time being                               
                minNode = self.PrepareMinNode(nearestIndex, xTrajs, trajCost)  
                # Get all the nodes in the DR-RRT* Tree that are closer to the randomNode within a specified search radius
                nearIndices = self.FindNearNodeIndices(randNode)                    
                # Choose the minimum cost path to connect the random node
                minNode = self.ConnectViaMinimumCostPath(nearestIndex, nearIndices,randNode, minNode)                               
                if int(minNode.means[-1, 0, 0]) != 0 or int(minNode.means[-1, 1, 0]) != 0:                    
                    # Add the minNode to the DR-RRT* Tree                
                    self.nodeList.append(minNode)  
                    # print('added new node to tree with cost:', minNode.cost)
                    # Rewire the tree with newly added minNode                    
                    self.ReWire(nearIndices, minNode) 
                # Quit building the tree once a node in goal region is successfully added to the tree
                if self.xmingoal <= minNode.means[-1,0,:] <= self.xmaxgoal and self.ymingoal <= minNode.means[-1,1,:] <= self.ymaxgoal:
                    goalNodeCounter = goalNodeCounter + 1
                    if goalNodeCounter > 2:
                        print('Reached goal region at iteration number:', iter)
                        break

###############################################################################
########################## FUNCTIONS CALLED BY MAIN() #########################
###############################################################################

def DefineStartParameters(planner_params): 

    # Unwrap all the planner parameters        
    start_w   = planner_params["start_w"]
    end_w     = planner_params["end_w"] 
    obs_w     = planner_params["obs_w"] 
    boundary  = planner_params["boundary"] # [bottom_left x, bottom_left y, wd, ht]
    steerLaw  = planner_params["steerLaw"]
    EgoCarDim = planner_params["EgoDim"]       
    maxIter   = planner_params["maxIter"] # Maximum number of iterations for tree building
    desDist   = planner_params["desDist"] # Desired Threshold distance
    envInfo   = planner_params["envInfo"]
    
    bottom_left_x   = boundary[0]
    bottom_left_y   = boundary[1]
    boundary_width  = boundary[2]
    boundary_height = boundary[3]
    bottom_right_x  = boundary[0] + boundary[2]
    bottom_right_y  = boundary[1]
    top_left_x      = boundary[0]
    top_left_y      = boundary[1] + boundary[3]
    top_right_x     = boundary[0] + boundary[2]
    top_right_y     = boundary[1] + boundary[3]
    
    xmin, ymin, xwd, yht = envInfo
    env_xmin, env_xmax, env_ymin, env_ymax = xmin, xmin+xwd, ymin, ymin+yht
    env_width = env_xmax-env_xmin
    env_height = env_ymax-env_ymin
    env_thickness = 1.0
    env_obstacles = [(env_xmin-env_thickness, env_ymin-env_thickness,                 
                      env_thickness, env_height+2*env_thickness),
                      (env_xmax, env_ymin-env_thickness, 
                       env_thickness, env_height+2*env_thickness),
                      (env_xmin-env_thickness, env_ymin-env_thickness, 
                       env_width+2*env_thickness, env_thickness),
                      (env_xmin-env_thickness, env_ymax, 
                       env_width+2*env_thickness, env_thickness)]
    
    # Infer the boundary conditions
    print('Displaying Boundary Information')
    print('(' + str(top_left_x) + ',' + str(top_left_y) + ')#######' + '(' + str(top_right_x) + ',' + str(top_right_y) + ')')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('(' + str(bottom_left_x) + ',' + str(bottom_left_y) + ')#########' + '(' + str(bottom_right_x) + ',' + str(bottom_right_y) + ')')
    
    # Fill environment details      
    start = [start_w[0], 
             start_w[1], 
             start_w[2], 
             start_w[3], 
             obs_w[0], 
             obs_w[1], 
             EgoCarDim[0], 
             EgoCarDim[1], 
             desDist] 
    goal = [end_w[0], 
            end_w[1]]    
    
    # randArea = [simArea_x_min, simArea_x_max, simArea_y_min, simArea_y_max]  # [xmin,xmax,ymin,ymax] 
    randArea = [bottom_left_x, bottom_right_x, top_left_y, bottom_left_y]  # [xmin,xmax,ymin,ymax] 
    print('Simulation Area', randArea)
    
    goalArea = [goal[0]-1, goal[0]+1, goal[1]-1, goal[1]+1]  # [xmin,xmax,ymin,ymax] 
    
    print('Goal Area', goalArea)
    
    # Infer the obstacle information
    obs_bl_x = obs_w[0] - (obs_w[3]/2) # Bottom-left car box x-pos
    obs_bl_y = obs_w[1] + (obs_w[4]/2) # Bottom-left car box y-pos
    obs_br_x = obs_w[0] + (obs_w[3]/2) # Bottom-right car box x-pos
    obs_br_y = obs_w[1] + (obs_w[4]/2) # Bottom-right car box y-pos
    obs_tl_x = obs_w[0] - (obs_w[3]/2) # Top-left car box x-pos
    obs_tl_y = obs_w[1] - (obs_w[4]/2) # Top-left car box y-pos
    obs_tr_x = obs_w[0] + (obs_w[3]/2) # Top-right car box x-pos
    obs_tr_y = obs_w[1] - (obs_w[4]/2) # Top-right car box y-pos
    
    
    print('Displaying Obstacle Information')
    print('(' + str(obs_tl_x) + ',' + str(obs_tl_y) + ')#######' + '(' + str(obs_tr_x) + ',' + str(obs_tr_y) + ')')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('###################################')
    print('(' + str(obs_bl_x) + ',' + str(obs_bl_y) + ')#########' + '(' + str(obs_br_x) + ',' + str(obs_br_y) + ')')
        
    
    # Obstacle Location Format [ox,oy,wd,ht]: 
    # (ox, oy):bottom left corner of rectangle, (wd):Width and (ht):height
    # Negate height argument for carla simulation
    obstacleList = [(obs_bl_x, obs_bl_y, obs_w[3], -obs_w[4])]
    print('Obstacle info (x,y,wd,ht)=', obstacleList[0])
    
    # Define the dynamics data - Car Robot Dynamics
    numStates   = steerLaw.controller.nmpc_params.num_states  # 6 # x,y,yaw,v,obx,oby
    numControls = steerLaw.controller.nmpc_params.num_ctrls   # 2 # accel, steer
    numOutputs  = steerLaw.controller.nmpc_params.num_outputs # 2 # r, phi        
        
    # Specify the initial state covariance
    S0 = steerLaw.controller.nmpc_params.SigmaE   
    
    # Define the covariances of process and sensor noises
    SigmaW = steerLaw.controller.nmpc_params.SigmaW # 0.05*np.identity(numStates)  # Covariance of process noise 
    SigmaV = steerLaw.controller.nmpc_params.SigmaV # 0.01*np.identity(numOutputs) # Covariance of sensor noise 
        
    return StartParams(start, 
                       randArea, 
                       goalArea, 
                       maxIter, 
                       obstacleList, 
                       env_obstacles,
                       Dynamics(numStates, numControls, numOutputs, S0, SigmaW, SigmaV))


###############################################################################
########################## MAIN() FUNCTION ####################################
###############################################################################

def Planner(planner_params):

    # Unwrap steerLaw from the planner parameters             
    steerLaw = planner_params["steerLaw"]
    
    # Prepare starting parameters
    startParam = DefineStartParameters(planner_params)
    
    # Create an object of DR_RRT_Star class
    dr_rrtstar = DR_RRTStar(startParam) 
    
    # starting time
    start = time.time()
    
    # Perform DR_RRTStar Tree Expansion
    dr_rrtstar.ExpandTree(steerLaw) 
    
    # end time
    end = time.time()
    
    # Print the time taken to compute the tree
    print('Time taken to develop motion plan:', round(end - start, 2), 'seconds')
    
    ######################## Save DR-RRTStar Tree Data ########################       
    # Pickle the nodeList data and dump it for further analysis
    if dr_rrtstar.DRFlag:
        filename = 'EKF_DR_RRT_Star_Tree.pkl'
    else:
        filename = 'CC_RRT_Star_Tree.pkl'
    outfile  = open(filename,'wb')
    pickle.dump(dr_rrtstar.nodeList, outfile)
    outfile.close()  
    
    # Display the final tree information
    print('Final Tree is')    
    fmt = '{:<8}{:<10}{:<10}{}' 
    print(fmt.format('Node_ID', 'x', 'y', 'Cost'))
    for k in range(len(dr_rrtstar.nodeList)):
        k_node = dr_rrtstar.nodeList[k]
        print(fmt.format(k, round(k_node.means[-1,0,:][0], 2), 
                          round(k_node.means[-1,1,:][0], 2),
                          round(k_node.cost, 2))) 
    
###############################################################################
###############################################################################
###################### END OF THE FILE ########################################
###############################################################################
###############################################################################