# Nonlinear Risk Bounded Robot Motion Planning
This code simulates the bicycle dynamics of car by steering it on the road by avoiding another static car obstacle in a CARLA simulator. The ego_vehicle has to consider all the system and perception uncertainties to generate a risk-bounded motion plan and execute it with coherent risk assessment. Coherent risk assessment for a nonlinear robot like the car in this simulation is made possible using nonlinear model predictive control (NMPC) based steering law combined with Unscented Kalman filter for state estimation purpose. Finally, distributionally robust chance constraints applied using a temporal logic specifications evaluate the risk of a trajectory before being added to the sequence of trajectories forming a motion plan from the start to the destination.

![Motion Planning Using Carla Simulator](https://github.com/venkatramanrenganathan/Risk_Bounded_Nonlinear_Robot_Motion_Planning/blob/main/carla_tree_path.PNG)

The code in this repository implements the algorithms and ideas from our following paper:
1. V. Renganathan, S. Safaoui, A. Kothari, I. Shames, T. Summers, `Risk Bounded Nonlinear Robot Motion Planning With Integrated Perception & Control`, Submitted to the International Journal on Robotics Research (IJRR), 2021.

# Dependencies
- Python 3.5+ (tested with 3.7.6)
- Numpy
- Scipy
- Matplotlib
- Casadi
- Namedlist
- Pickle
- Carla


# Installing
You will need the following two items to run the codes. After that there is no other formal package installation procedure; simply download this repository and run the Python files.
- CARLA SIMULATOR VERSION: 0.9.10
- UNREAL ENGINE VERSION: 4.24.3

# Modules of an autonomy stack
There are two main modules for understanding this whole package
1. First, a high level motion planner has to run and it will generate a reference trajectory for the car from start to the end
2. Second, a low level tracking controller will enable the car to track the reference trajectory despite the realized noises.

# Procedure to run the code
1. Run the python code `Generate_Monte_Carlo_Noises.py` which will generate and load the required noise parameters and data required for simulation into pickle files
2. Run the python code `Run_Path_Planner.py`
3. The code will run for specified number of iterations and produces all required data
4. Then load the cooresponding pickle file data in file `main.py` in the line number #488.
5. Run the `main.py` file with the Carla executable being open already
6. The simulation will run in the Carla simulator where the car will track the reference trajectory and results are stored in pickle files
7. To see the tracking results, run the python file `Tracked_Path_Plotter.py`

# Running Monte-Carlo Simulations

1. Create a new folder called `monte_carlo_results` in the same directory where the python file `monte_carlo_car.py` resides.
1. Update the `trial_num` at line #1554 in the file `monte_carlo_car.py` and run it while the Carla executable is open (It will automatically load the noise realizations corresponding to the `trial_num` from the pickle files)
1. After the simulation is over, automatically the results are stored under the folder `monte_carlo_results` with a specific trial name
1. Repeat the process by changing trial number in step 2 and run again.
1. Once the all trials are completed, run the python file `monte_carlo_results_plotter.py` to plot the monte-carlo simulation results 

# Variations
- Instead of Distributionally robust chance constraints, if you would like to have a simple Gaussian Chance Constraints, then change 
```self.DRFlag = False``` in line 852 in the file `DR_RRTStar_Planner.py`
- Choose your own state estimator UKF or EKF by commenting and uncommenting the corresponding estimator in lines 26-27 of file `State_Estimator.py`

# Funding Acknowledgement
This work is partially supported by *Defence Science and Technology Group*, through agreement MyIP: ID10266 entitled **Hierarchical Verification of Autonomy Architectures**, the Australian Government, via grant `AUSMURIB000001` associated with ONR MURI grant `N00014-19-1-2571`, and by the *United States Air Force Office of Scientific Research* under award number `FA2386-19-1-4073`.

# Contributing Authors
1. [Venkatraman Renganathan - UT Dallas](https://github.com/venkatramanrenganathan)
2. [Sleiman Safaoui - UT Dallas](https://github.com/The-SS)
3. [Aadi Kothari - UT Dallas](https://github.com/Aadi0902)
4. [Tyler Summers - UT Dallas](https://github.com/TSummersLab)

# Affiliation
[TSummersLab - Control, Optimization & Networks Laboratory (CONLab)](https://github.com/TSummersLab)
