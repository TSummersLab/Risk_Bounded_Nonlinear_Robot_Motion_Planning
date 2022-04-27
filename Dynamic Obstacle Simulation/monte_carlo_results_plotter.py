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
import os
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import config
SAVEPATH = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), 'monte_carlo_results')  # path to save data

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
    filename = 'pathNodesList.pkl'
    infile = open(filename,'rb')
    pathNodesList = pickle.load(infile)
    infile.close()     
    
    # Unbox Pickle file to load the waypts_ref_data
    filename = 'waypts_ref_data.pkl'
    infile = open(filename,'rb')
    waypts_ref_data = pickle.load(infile)
    infile.close() 
    
    # Unbox Pickle file to load the nodeList data 
    car_tracking_data_list = []    
    trial_num = config.Trial_Num
    for k in range(trial_num):    
        file_name = "mc_results_"+str(k+1)+'.pkl'
        infile = open(os.path.join(SAVEPATH, file_name),"rb")
        car_tracking_data_k = pickle.load(infile)
        car_tracking_data_list.append(car_tracking_data_k)
        infile.close()    
    
    
    # Plot the data
    fig = plt.figure(figsize = [16,9])
    # create an axes object in the figure
    ax = fig.add_subplot(1, 1, 1)    
    
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
                                angle=0, color='k')
                      ]
    for obstacle in env_obstacles:
        ax.add_artist(obstacle)
        
    # Plot car image as obstacle
    # arr_car = mpimg.imread('car.jpg')
    # imagebox = OffsetImage(arr_car, zoom=0.05)    
    # ab = AnnotationBbox(imagebox, (obst[0], obst[1]+1))    
    # ax.add_artist(ab)
        
    
    # Plot the rectangle obstacles
    ax.add_artist(Rectangle((obst[0]-obst[3]/2, obst[1]-obst[4]/2), obst[3], obst[4], lw=2, fc='k', ec='k', alpha = 0.3))
    # Plot goal region
    ax.add_artist(Rectangle((end_w[0]-1, end_w[1]-1), 2, 2, lw=2, ls = '--', fc='b', ec='b', alpha = 0.3))    
    # Plot start point
    ax.scatter(start_w[0], start_w[1], s=600, c='goldenrod', marker='^', edgecolor='k', label='Start', zorder=20, alpha = 0.8)
    
    
    ######################## Plot Reference Trajectory ########################
    # Plot the data
    xrefPtValues = [] 
    yrefPtValues = []     
    # Store to Plot RRT* node
    for refNode in pathNodesList:        
        xrefPtValues.append(refNode.means[-1,0,0])  
        yrefPtValues.append(refNode.means[-1,1,0])            
    plt.scatter(xrefPtValues, yrefPtValues, c = 'red', marker='o')
    # Marker for Goal Point
    plt.scatter(xrefPtValues[-1], yrefPtValues[-1], 
                s=800, c='goldenrod', marker='*', 
                edgecolor='k', linewidths=2, label='Goal', zorder=20)               
    # Plot the complete node sequence
    for i in range(waypts_ref_data.shape[1]-1):
        xPtRef = [waypts_ref_data[0,i], waypts_ref_data[0,i+1]]
        yPtRef = [waypts_ref_data[1,i], waypts_ref_data[1,i+1]]
        plt.plot(xPtRef, yPtRef, color = 'blue', linewidth = 4.0)            
    
    ######################## Plot Tracked Trajectory ##########################                
    # Plot the complete node sequence
    for k in range(len(car_tracking_data_list)):
        car_tracking_data_k = car_tracking_data_list[k]
        xmontecarlo = []
        ymontecarlo = []
        for i in range(car_tracking_data_k.shape[1]-1):
            xmontecarlo.append(car_tracking_data_k[0,i])
            xmontecarlo.append(car_tracking_data_k[0,i+1])
            ymontecarlo.append(car_tracking_data_k[1,i])
            ymontecarlo.append(car_tracking_data_k[1,i+1])            
        plt.plot(xmontecarlo, ymontecarlo, color = 'green', linewidth = 2.0, alpha = 0.3)            

    ############################ Cosmetic Code ################################
    # Keep equal aspect and invert y axis  
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
    boundaryInfo = [-83, -42.57, 10, -42.5]
    
    # Plot the data
    Plotter(start_w, end_w, obs_w, boundaryInfo)

###############################################################################
###############################################################################