# Dynamic Obstacle Based Simulation
This code simulates nonlinear risk bounded robot motion planning with dynamic obstacle. Specifically, a bicycle robot is made to avoid a moving obstacle that has a constant velocity of 0.1.  

## Procedure to run the simulation
1. Define the velocity of the obstacle by setting the velocitySelector flag in config.py file
2. Run the Run_Path_Planner.py file to build the tree and the reference path from the source to goal. This will generate all the required pickle file data necessary for plotting the results.
3. Run the NMPC_UKF_Path_Tracking_DynObstacle.py file. This will generate the simulation where the ego vehicle car overtaking the moving obstacle. 
4. Repeat steps 1-3 with different velocities.
5. Run Plot_Dynamic_Obstacle_Comparison.py file to compare the results of reference trajectories with different obstacle velocities. 
