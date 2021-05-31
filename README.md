# Risk_Bounded_Nonlinear_Robot_Motion_Planning
This code simulates the bicycle dynamics of car by steering it on the road by avoiding another static car obstacle. The ego_vehicle has to consider all the system and perception uncertainties to generate a risk-bounded motion plan and execute it with coherent risk assessment. This code uses the CARLA simulator.

- CARLA SIMULATOR VERSION - 0.9.10
- PYTHON VERSION          - 3.7.6
- VISUAL STUDIO VERSION   - 2017
- UNREAL ENGINE VERSION   - 4.24.3

![truck_platoon](https://github.com/venkatramanrenganathan/Attack-Resiliency-In-Truck-Platooning-Using-Two-Team-Games/blob/master/truck_platoon.gif)

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
Currently there is no formal package installation procedure; simply download this repository and run the Python files.

# Modules of an autonomy stack
There are two main modules for understanding this whole package
1. First, a high level motion planner has to run and it will generate a reference trajectory for the car from start to the end
2. Second, a low level tracking controller will enable the car to track the reference trajectory despite the realized noises.

# Procedure to run the code
1. Run the python code `Run_Path_Planner.py`
2. The code will run for specified number of iterations and produces all required data
3. Then load the cooresponding pickle file data in file `main.py` in the line number #488.
4. Run the `main.py` file with the Carla executable being open already
5. The simulation will run in the Carla simulator where the car will track the reference trajectory and results are stored in pickle files
6. To see the tracking results, run the python file `Tracked_Path_Plotter.py`

# Variations
- Instead of Distributionally robust chance constraints, if you need to have a simple Gaussian Chance Constraints, then change `self.DRFlag` as False in line 852 in the file `DR_RRTStar_Planner.py`
- Choose your own state estimator UKF or EKF by commenting and uncommenting the corresponding estimator in lines 26-27 of file `State_Estimator.py`

# Contributing Authors
1. [Venkatraman Renganathan - UT Dallas](https://github.com/venkatramanrenganathan)
2. [Sleiman Safaoui - UT Dallas](https://github.com/The-SS)
3. [Aadi Kothari - UT Dallas](https://github.com/Aadi0902)
4. [Tyler Summers - UT Dallas](https://github.com/TSummersLab)

# Affiliation
[TSummersLab - Control, Optimization & Networks Laboratory (CONLab)](https://github.com/TSummersLab)
