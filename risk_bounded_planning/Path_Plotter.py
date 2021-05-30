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
import math
from numpy import linalg as LA
from matplotlib.patches import Rectangle
from matplotlib.collections import EllipseCollection

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

def Plotter(start_w, end_w, obst, boundaryInfo):

    # Unbox Pickle file to load the nodeList data 
    filename = 'CC_RRT_Star_Tree.pkl'
    infile = open(filename,'rb')
    nodeList = pickle.load(infile)
    infile.close()
    
    # Unbox Pickle file to load the nodeList data 
    filename = 'EKF_pathNodesList.pkl'
    infile = open(filename,'rb')
    pathNodesList = pickle.load(infile)
    infile.close()     
    
    # ☺Unbox Pickle file to load the waypts_ref_data
    filename = 'EKF_waypts_ref_data.pkl'
    infile = open(filename,'rb')
    waypts_ref_data = pickle.load(infile)
    infile.close()        
    
    # Unbox Pickle file to load the queried points
    filename = 'ekf_freePoints.pkl'
    infile = open(filename,'rb')
    freePoints = pickle.load(infile)
    xFreePoints, yFreePoints = freePoints
    infile.close()        
    
    # Plot the data
    fig = plt.figure(figsize = [16,9])
    # create an axes object in the figure
    ax = fig.add_subplot(1, 1, 1) 
    # Plot the queried points
    # ax.scatter(xFreePoints, yFreePoints, s=20, c='black', marker='x', label='Sampled Points', zorder=20)
    
    # Plot the environment boundary obstacles
    xmin, ymin, xwd, yht = boundaryInfo
    env_xmin, env_xmax, env_ymin, env_ymax = xmin, xmin+xwd, ymin, ymin+yht
    env_width = env_xmax-env_xmin
    env_height = env_ymax-env_ymin
    env_thickness = 1.0
    env_obstacles = [Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                                width=env_thickness,
                                height=env_height+2*env_thickness,
                                angle=0, color='k'),
                      Rectangle(xy=(env_xmax, env_ymin-env_thickness),
                                width=env_thickness,
                                height=env_height+2*env_thickness,
                                angle=0, color='k'),
                      Rectangle(xy=(env_xmin-env_thickness, env_ymin-env_thickness),
                                width=env_width+2*env_thickness,
                                height=env_thickness,
                                angle=0, color='k'),
                      Rectangle(xy=(env_xmin-env_thickness, env_ymax),
                                width=env_width+2*env_thickness,
                                height=env_thickness,
                                angle=0, color='k')]
    for obstacle in env_obstacles:
        ax.add_artist(obstacle)
    
    
    # Plot the rectangle obstacles - obst = [-75, -65, 1.02, 2.49]
    obst_rect = Rectangle((obst[0]-obst[3]/2, obst[1]-obst[4]/2), obst[3], obst[4], lw=2, fc='k', ec='k', alpha = 0.5, label='Obstacle')
    ax.add_artist(obst_rect)
    # Plot goal region
    ax.add_artist(Rectangle((end_w[0]-1, end_w[1]-1), 2, 2 , lw=2, fc='b', ec='b', alpha = 0.3))    
    # Plot start point
    ax.scatter(start_w[0], start_w[1], s=400, c='goldenrod', ec= 'k', linewidths=2, marker='^', label='Start', zorder=20, alpha = 0.3)
    
    # Plot the DR-RRT* Tree
    xPtValues = []
    yPtValues = [] 
    xValues = []
    yValues = []
    widthValues = []
    heightValues = []
    angleValues = []
    for k, ellipseNode in enumerate(nodeList):
        # if k == 67:
        #     continue
        xPtValues.append(ellipseNode.means[-1,0,0])
        yPtValues.append(ellipseNode.means[-1,1,0])
        if ellipseNode is not None and ellipseNode.parent is not None:                
            ellNodeShape = ellipseNode.means.shape  
            xPlotValues  = []
            yPlotValues  = []            
            # Prepare the trajectory x and y vectors and plot them                
            for k in range(ellNodeShape[0]):                                    
                xPlotValues.append(ellipseNode.means[k,0,0])
                yPlotValues.append(ellipseNode.means[k,1,0]) 
            # Plotting the risk bounded trajectories
            plt.plot(xPlotValues, yPlotValues, color = '#636D97', linewidth = 1.0)
            # Plot only the last ellipse in the trajectory
            alfa = math.atan2(ellipseNode.means[-1, 1, 0], ellipseNode.means[-1, 0, 0])
            xValues.append(ellipseNode.means[-1, 0, 0])
            yValues.append(ellipseNode.means[-1, 1, 0])
            elcovar  = np.asarray(ellipseNode.covar[-1,:,:])            
            elE, elV = LA.eig(elcovar[0:2,0:2])
            widthValues.append(2*math.sqrt(elE[0]))
            heightValues.append(2*math.sqrt(elE[1]))
            angleValues.append(alfa*360)
    plt.scatter(xPtValues, yPtValues, c = '#e87500', marker='o')
    XY = np.column_stack((xValues, yValues))
    ec = EllipseCollection(widthValues,
                           heightValues,
                           angleValues,
                           units='x',
                           offsets=XY,
                           facecolors="#C59434",
                           transOffset=ax.transData)
    ec.set_alpha(0.6)
    ax.add_collection(ec)
    
    #print('badnodes', nodeList[33].means)
    
    ######################## Plot Reference Trajectory ########################
    #if False:
    if pathNodesList is not None:
        # Plot the data
        xrefPtValues = [] 
        yrefPtValues = []     
        # Store to Plot RRT* node
        for refNode in pathNodesList:          
            xrefPtValues.append(refNode.means[-1,0,0])
            yrefPtValues.append(refNode.means[-1,1,0])            
        plt.scatter(xrefPtValues, yrefPtValues, c = 'blue', marker='o', s=60)    
        # Marker for Goal Point
        plt.scatter(xrefPtValues[-1], yrefPtValues[-1], 
                    s=600, c='goldenrod', marker='*', 
                    edgecolor='k', linewidths=2, label='Goal', zorder=20)        
        # Plot the complete node sequence
        for i in range(waypts_ref_data.shape[1]-1):
            xPtRef = [waypts_ref_data[0,i], waypts_ref_data[0,i+1]]
            yPtRef = [waypts_ref_data[1,i], waypts_ref_data[1,i+1]]
            plt.plot(xPtRef, yPtRef, color = '#0078f0', linewidth = 4.0, ls='dashed')            
    
    ############################ Cosmetic Code ################################
    # Keep equal aspect and invert y axis # plt.ylim((-85,-40)) # plt.xlim((-85,-70))        
    ax.axis('equal')
    ax.axis([-85,-70,-85,-40])
    ax.axis('off')     
    ax.set_xlabel('x')
    ax.set_ylabel('y')        
    ax.autoscale(False)
    ax.invert_yaxis() 
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.legend().get_texts()):
        item.set_fontsize(20)

###############################################################################
###############################################################################

if __name__ == '__main__':

    # Ego Vehicle    
    start_w = [-75.0, -45.0]    
    # Goal
    end_w = [-75.0, -75.0]    
    # Infer the position of obstacle
    obs_w = [-75.0, -65.0, Wrap_Angle(180), 1.02, 2.49]
    # Infer the boundary information
    boundaryInfo = [-83, -42.5, 10, -42.5]  # [-81-1.02=-82.02, -42.43-2.43--44.86, 7+1.02=-8, -45-2.5=-47.5]
    
    # Plot the data
    Plotter(start_w, end_w, obs_w, boundaryInfo)

###############################################################################
###############################################################################