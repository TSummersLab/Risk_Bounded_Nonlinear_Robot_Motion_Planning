# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:01:02 2021

@author: vxr131730
"""
import numpy as np
import pickle

num_states = 6
num_outputs = 4
Trial_Total = 5000
mu_w = np.zeros(num_states)
mu_v = np.zeros(num_outputs) 
joint_mu = np.append(mu_w, mu_v)
joint_dim = num_states + num_outputs
# while True:
#     joint_Cov = np.random.rand(joint_dim,joint_dim)
#     joint_Cov = 0.5*(joint_Cov + joint_Cov.T)
#     joint_Cov = joint_Cov + (joint_dim*np.eye(joint_dim))
#     joint_Cov = 0.0001*joint_Cov
#     if np.all(np.linalg.eigvals(joint_Cov) > 0):
#         break
joint_Cov = 0.0000001*np.eye(joint_dim)
SigmaW = joint_Cov[0:num_states, 0:num_states]
SigmaV = joint_Cov[num_states:, num_states:]
CrossCor = joint_Cov[0:num_states, num_states:] 

# Generate independent Noises across all trials
WV = np.random.multivariate_normal(joint_mu, joint_Cov, Trial_Total).T

# Prepare output to be dumped into pickle file
noise_data = [joint_Cov, WV]

# Save pathNodesList to pickle 
print('Saving noise data as pickle file')
filename = 'noise_data.pkl'
outfile  = open(filename,'wb')
pickle.dump(noise_data, outfile)
outfile.close()



