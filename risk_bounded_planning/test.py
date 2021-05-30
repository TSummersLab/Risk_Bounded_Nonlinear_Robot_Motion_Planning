# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:34:06 2021

@author: vxr131730
"""

import os
import pickle
SAVEPATH = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), 'monte_carlo_results')  # path to save data
MC_FOLDER = os.path.join('..', 'monte_carlo_results')

# Unbox Pickle file to load the nodeList data 
filename = 'waypts_ref_data'
infile = open(filename,'rb')
waypts_ref_data = pickle.load(infile)
infile.close()  

# Save in monte-carlo folder
trial_num = 3
file_name = "mc_results_"+str(trial_num)+'.pkl'
pickle_on = open(os.path.join(SAVEPATH, file_name),"wb")
pickle.dump(waypts_ref_data, pickle_on)
pickle_on.close()







# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.patches import Rectangle


# left, width = 76, 2.5
# bottom, height = 64, 1.02
# right = left + width
# top = bottom + height

# fig,ax = plt.subplots(1)

# # axes coordinates are 0,0 is bottom left and 1,1 is upper right
# p = Rectangle((left, bottom), width, height, 
#               linewidth=1,edgecolor='r',facecolor='none', fill=True)
# ax.add_artist(p)
# plt.xlim([left-1, right+1])
# plt.ylim([bottom-1, top+1])



# # Plot the data
# fig = plt.figure(figsize = [16,9])
# ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
# # Plot the rectangle obstacles       
# obstacles = patches.Rectangle(xy = [76.245, 64.49], 
#                       width  = 2.49, 
#                       height = 1.02, 
#                       angle  = 0, 
#                       color  = "k")
# ax.add_patch(obstacles)
# # ax.add_artist(obstacles) 
# plt.show()    