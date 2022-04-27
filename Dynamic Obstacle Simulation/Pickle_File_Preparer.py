# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:28:07 2021

@author: vxr131730
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:00:48 2021

@author: vxr131730
"""
###############################################################################
###############################################################################

# Import all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import config

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

def Wrap_Angle(angle_in_degree):
    """
    This function converts the angle input values in degree to radian values
    between [-pi, pi] and returns the resulting radian angle value

    Parameters
    ----------
    angle_in_degree : FLOAT
        Angle value in degrees.

    Returns
    -------
    angle_in_rad : FLOAT
        Angle value in radians.

    """
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while angle_in_rad > np.pi:
        angle_in_rad -= 2*np.pi
    while angle_in_rad <= -np.pi:
        angle_in_rad += 2*np.pi
    return angle_in_rad    

###############################################################################
###############################################################################    

def GenerateReferencePath(nodeList, xGoal, yGoal): 
    
    # Just set goalIndex to be zero
    goalIndex = 0
    goalIndices = []
    # Get the index of the node in the goal area
    for node in nodeList:            
        if (node.means[-1,0,:] >= xGoal - 1 and 
            node.means[-1,1,:] >= yGoal - 1 and 
            node.means[-1,0,:] <= xGoal + 1 and 
            node.means[-1,1,:] <= yGoal + 1):  
            goalIndices.append(nodeList.index(node))           
            # goalIndex = nodeList.index(node)           
            # break
    
    if goalIndices:
        goalIndex = random.choice(goalIndices)
    print('Index of Goal Node is:', goalIndex)
    
    # If no goalindex is found, return an empty list
    if goalIndex == 0:
        pathNodesList = []
        return pathNodesList
    
    # Initialize the reference node list
    pathNodesList = [nodeList[goalIndex]]
    
    # Loop until the root node (whose parent is None) is reached
    while nodeList[goalIndex].parent is not None:            
        # Set the index to its parent
        goalIndex = nodeList[goalIndex].parent
        # Append the parent node to the pathNodeList
        pathNodesList.append(nodeList[goalIndex])
        
    # Finally append the path with root node        
    pathNodesList.append(nodeList[0])  

    # Currently the list is from goal to root - Reverse it from root to goal
    pathNodesList = pathNodesList[::-1]      
    
    return pathNodesList    

###############################################################################
###############################################################################

def Save_Data(xGoal, yGoal):

    # Unbox Pickle file to load the nodeList data     
    if config.DRFlag:
        if config.estimatorSelector:
            # DR-UKF
            if config.velocitySelector == 0:
                filename = 'DR_RRT_Star_Tree010.pkl'
            if config.velocitySelector == 1:
                filename = 'DR_RRT_Star_Tree020.pkl'
        else:
            # DR-EKF
            filename = 'DR_EKF_RRT_Star_Tree.pkl'
    else:
        # CC-UKF
        filename = 'CC_RRT_Star_Tree.pkl'
    infile = open(filename,'rb')
    nodeList = pickle.load(infile)
    infile.close()
    
    # Generate the sampled path from goal node to root node
    pathNodesList = GenerateReferencePath(nodeList, xGoal, yGoal)   
    
    # Save pathNodesList to pickle 
    if config.DRFlag:
        if config.estimatorSelector:
            # DR-UKF
            filename = 'pathNodesList.pkl'
        else:
            # DR-EKF
            filename = 'ekf_pathNodesList.pkl'
    else:
        # CC-UKF
        filename = 'cc_pathNodesList.pkl'    
    outfile  = open(filename,'wb')
    pickle.dump(pathNodesList,outfile)
    outfile.close()  
    
    ###########################################################################
    # Prepare Waypoints Data from DR-RRT* reference nodes
    ########################################################################### 
    
    # Save the data    
    if pathNodesList is not None:    
        xPtVals   = [] 
        yPtVals   = []
        yawPtVals = []
        vPtVals   = []
        obxPtVals = []
        obyPtVals = []      
        for refNode in pathNodesList:               
            # Prepare the trajectory x and y vectors and plot them                
            for k in range(refNode.means.shape[0]):                                    
                xPtVals.append(refNode.means[k,0,0])
                yPtVals.append(refNode.means[k,1,0])
                yawPtVals.append(refNode.means[k,2,0])
                vPtVals.append(refNode.means[k,3,0])
                obxPtVals.append(refNode.means[k,4,0])
                obyPtVals.append(refNode.means[k,5,0])            
        waypts_ref_data = np.array((xPtVals, yPtVals, yawPtVals, vPtVals, obxPtVals, obyPtVals))
    else:
        print('Unable to generate reference trajectory - Sample more points !!!')
        waypts_ref_data = []
        
    # Duplicate the last column N times for perfect tracking (N = 6 is tracking MPC horizon)
    # Suppose we have the matrix waypts_ref_data
    # A = [1,2,3  Then we need B = [1,2,3,3
    #      4,5,6                    4,5,6,6  
    #      7,8,9]                   7,8,9,9'
    #                              
    duplicateIndex = 100
    if not waypts_ref_data.all():                            
        waypts_ref_data = np.hstack((waypts_ref_data, np.tile(waypts_ref_data[:, [-1]], duplicateIndex)))
        for i in range(1,duplicateIndex-1):
            waypts_ref_data[0,-i] = xGoal
            waypts_ref_data[1,-i] = yGoal-0.1
        
    
    # Pickle the nodeList data and dump it for further analysis
    if config.DRFlag:
        if config.estimatorSelector:
            # DR-UKF
            if config.obsVelocities[config.velocitySelector] == 0.10:
                filename = 'waypts_ref_data010.pkl'
            if config.obsVelocities[config.velocitySelector] == 0.20:
                filename = 'waypts_ref_data020.pkl'
        else:
            # DR-EKF
            if config.obsVelocities[config.velocitySelector] == 0.10:
                filename = 'ekf_waypts_ref_data010.pkl'
            if config.obsVelocities[config.velocitySelector] == 0.20:
                filename = 'ekf_waypts_ref_data020.pkl'
    else:
        # CC-UKF
        if config.obsVelocities[config.velocitySelector] == 0.10:
            filename = 'cc_waypts_ref_data010.pkl'
        if config.obsVelocities[config.velocitySelector] == 0.20:
            filename = 'cc_waypts_ref_data020.pkl'
    outfile  = open(filename,'wb')
    pickle.dump(waypts_ref_data, outfile)
    outfile.close()    

    print('Finished Saving All Data')

###############################################################################
###############################################################################

if __name__ == '__main__':
    
    xGoal = -75.0
    yGoal = -75.0
    
    # Plot the data
    Save_Data(xGoal, yGoal)

###############################################################################
###############################################################################