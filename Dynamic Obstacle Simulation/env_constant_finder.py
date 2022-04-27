# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:25:29 2021

@author: vxr131730
"""

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def ComputeLyapunovDistance(fromNode, toNode):
    """
    Returns the distance between two nodes computed using the dynamic control-based distance metric
    Input parameters:
    fromNode   : Node representing point A
    toNode     : Node representing point B        
    """
    
    # Use the dynamic control-based distance metric
    diffVec = np.array(fromNode) - np.array(toNode)
    
    # Define the scaling constants for the robot
    kPhi = 1.2
    kDel = 3  
    
    # Compute the radial (Euclidean) target distance
    r = LA.norm(diffVec)        
    
    # Orientation of the vehicle heading w.r.t LOS - Called Del in (-pi, pi]
    Del = -2*np.pi + math.atan2(toNode[1], toNode[0])
    
    # Orientation of target pose w.r.t. LOS - Called Phi in (-pi, pi]
    Phi = toNode[2] + Del
    
    # desired heading points directly to target pose
    dDel = 0   
    
    # Compute the distance
    dist = math.sqrt(r**2 + (kPhi*Phi)**2) + kDel*abs(Del - dDel)

    return dist       

def PointsInCircum(r, n):
    return [(math.cos(2*np.pi/n*x)*r,math.sin(2*np.pi/n*x)*r) for x in range(0,n+1)]    

# # Environmental constant finder
# d = 2 # dimension 
# ENVCONSTANT = 30   
# searchRadii = []

# for totalNodes in range(1,1000):    
#     searchRadii.append(ENVCONSTANT*(math.log(totalNodes)/(totalNodes))**(1/(d+1)))

# fig, ax = plt.subplots(figsize=(3, 3))
# plt.plot(searchRadii)


# Threshold distance finder
fromNode = [0, 0, 0]
radius = 5
numpts = 10
toNodes = PointsInCircum(radius, numpts)
print(toNodes)
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(fromNode[0], fromNode[1])
dists = []
for i in range(len(toNodes)):
    toNode = list(toNodes[i])
    toNode.append(0)
    dists.append(ComputeLyapunovDistance(fromNode, toNode))
    ax.scatter(toNode[0], toNode[1])
ax.plot(dists)
    
fmt = '{:<10}{:<10}{}' 
print(fmt.format('x', 'y', 'Cost'))
for i in range(len(toNodes)):
    toNode = list(toNodes[i])
    print(fmt.format(round(toNode[0], 2), 
                     round(toNode[1], 2),
                     round(dists[i], 2))) 
minindex = dists.index(min(dists))
maxindex = dists.index(max(dists))
print('Least distance is', round(dists[minindex],2))
print('Least distant point is', toNodes[minindex])
print('Farthest distance is', round(dists[maxindex],2))
print('Farthest distant point is', toNodes[maxindex])

