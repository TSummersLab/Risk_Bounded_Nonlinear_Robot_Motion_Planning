# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:12:12 2020

@author: vxr131730
"""

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
from test import *
from casadi import *
from numpy import random as npr
from casadi.tools import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla

IM_WIDTH = 640
IM_HEIGHT = 480



actor_list = []

try:
    client = carla.Client("localhost",2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    startpoint = world.get_map().get_spawn_points()[195] #128 195
    world.debug.draw_string(startpoint.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=50,
                                        persistent_lines=True)

    vehicle = world.spawn_actor(vehicle_bp, startpoint)
    # ---------------------Trajectory-----------------
    wplist = world.get_map().get_topology()
    wps = wplist[270][0].next_until_lane_end(5.0) # 22 (195:270)
    for w in wps:
        world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                    color=carla.Color(r=0, g=255, b=0), life_time=20.0,
                                    persistent_lines=True)    
    endpoint = wps[0].transform
    actor_list.append(vehicle)

    # -------------------MPC--------------------

    # SAMPLING TIME
    T = 0.08 #0.08 #  (s)

    #  PREDICTION HORIZON
    N = 50 # 12 5

    # STATES
    x = SX.sym('x') # x coordinate
    y = SX.sym('y') # y coordinate
    theta = SX.sym('theta') # vehicle orientation
    v = SX.sym('v') # longitudenal velocity
    states = vertcat(x,y,theta,v)
    n_states = 4 # no. of states

    # CONTROL
    thr = SX.sym('thr') # Throttle
    strang = SX.sym('strang') # steering angle
    controls = vertcat(thr,strang)
    n_controls = 2

    # CONTROL BOUNDS
    minthr = 0.0 # minimum throttle
    maxthr = 0.5 # maximum throttle
    minstrang = -1 # minimum steering angle 
    maxstrang = 1 # maximum steering angle

    # VEHICLE MODEL PARAMETERS
    l_r = 1.415 # distance from center of gravity to rare wheels
    l_f = 1.6# distance from center of graity to front wheels

    # SYMBOLIC REPRESENTATION OF THE DERIVATIVE OF THE STATES BASED ON THE BICYCE MODEL
    rhs = vertcat((v*cos(theta+(atan((l_r*tan(strang*1.22))/(l_f + l_r))))), 
                  v*sin(theta+(atan((l_r*tan(strang*1.22))/(l_f + l_r)))), 
                  ((v/l_r)*(sin(atan((l_r*tan(strang*1.22))/(l_f + l_r))))), 
                  thr*16)
    
    # STATE PREDICTION - SYMBOLIC FUNCTION OF CURRENT STATE AND CONTROL INPUTS AT EACH TIME STEP OF HORIZON PERIOD
    f = Function('f', [states, controls], [rhs])
    U = SX.sym('U', n_controls, N)
    P = SX.sym('P', n_states + n_states)
    X = SX.sym('X', n_states, (N+1))
    X[:,0] = P[0:4]

    for k in range(N):
        st = X[:,k]
        con = U[:,k]
        f_value = f(st, con)
        st_next = st + (T*f_value)
        X[:,k+1] = st_next
    ff = Function('ff', [U,P], [X])
    
    # SYBOLIC REPRESENTATION OF THE OBJECTIVE FUNCTION
    obj = 0
    g = SX.sym('g',4,(N+1))
    Q = diag(SX([3600,3600,1900,2])) #195 [3600,3600,1900,2] [3100,3100,1900,2]  [2700,2700,2000,2]
    R = diag(SX([0,8000])) #195  [0,7000]

    for k in range(N):
        st = X[:,k]
        con = U[:,k]
        obj = obj + mtimes(mtimes((st - P[4:8]).T,Q), (st - P[4:8])) + mtimes(mtimes(con.T, R), con)
    
    # STATES BOUNDS/CONSTRAINTS
    for k in range(0,N+1):
        g[0,k] = X[0,k]
        g[1,k] = X[1,k]
        g[2,k] = X[2,k] 
        g[3,k] = X[3,k]
    g = reshape(g, 4*(N+1), 1)

    # CREATING A OPTIMIZATION SOLVER IN CASADI
    OPT_variables = reshape(U, 2*N, 1)

    nlp_prob = {'f':obj, 'x':OPT_variables, 'g':g, 'p':P}

    opts = {'ipopt.max_iter':100, 
            'ipopt.print_level':0, 
            'print_time':0, 
            'ipopt.acceptable_tol':1e-8, 
            'ipopt.acceptable_obj_change_tol':1e-6}

    solver = nlpsol('solver','ipopt', nlp_prob, opts) # solver

    # IMPLEMENTING CONTROL BOUNDS
    lbx = []
    ubx = []
    for i in range(2*N):
        if i%2==0:
            lbx.append(minthr)
            ubx.append(maxthr)
        else:
            lbx.append(minstrang)
            ubx.append(maxstrang)
    lbx = np.transpose(lbx)
    ubx = np.transpose(ubx)

    # IMPLEMENTING STATE BOUNDS
    lbgv = []
    ubgv = []
    for i in range(0,4*(N+1),4):
        lbgv.append(-300)
        lbgv.append(-300)
        lbgv.append(0)
        lbgv.append(0)
        ubgv.append(300)
        ubgv.append(300)
        ubgv.append(405)
        ubgv.append(15)
    u0 = (DM.zeros(2*N,1))
    u_cl = []

    def contheta(thet):
        if thet < 0:
            thet = 360 - abs(thet)
        return thet

    x0 = np.transpose([startpoint.location.x, startpoint.location.y, contheta(startpoint.rotation.yaw), 0])
    xs = np.transpose([endpoint.location.x, endpoint.location.y, contheta(startpoint.rotation.yaw), 3]) #-90.156235*pi/180

    c = 0
    p = np.transpose([startpoint.location.x, 
                      startpoint.location.y, 
                      contheta(startpoint.rotation.yaw), 
                      0,
                      endpoint.location.x, 
                      endpoint.location.y, 
                      contheta(startpoint.rotation.yaw), 
                      3])    
    while c < len(wps):        
        if (norm_2(x0[0:2]-p[4:6]))<3:
            c += 1
            endpoint = wps[c].transform
            world.debug.draw_string(endpoint.location, 'O', draw_shadow=False,
                                                    color=carla.Color(r=0, g=0, b=255), life_time=3,
                                                    persistent_lines=True)
        print(x0,"---",p[4:8])
        u0 = reshape(u0, 2*N,1)
        p[0:4] = x0
        p[4:8] = [endpoint.location.x, endpoint.location.y, contheta(endpoint.rotation.yaw), 3]#6
        sol = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbgv, ubg=ubgv, p=p)
        u = reshape(sol['x'].T, 2, N).T
        ff_value = ff(u.T, p)
        for k in range(N):            
            world.debug.draw_string(carla.Location(x=float(ff_value[0,k]), 
                                                   y=float(ff_value[1,k]), 
                                                   z=0.0), 
                                    'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=0.01,
                                    persistent_lines=True)        
        u_cl.append(u[0,:])        
        vehicle.apply_control(carla.VehicleControl(throttle =float(u[0,0]) , steer = float(u[0,1])))
        u_theta = vehicle.get_transform().rotation.yaw        
        x0 = np.transpose([vehicle.get_transform().location.x, 
                           vehicle.get_transform().location.y, 
                           contheta(u_theta), 
                           norm_2([vehicle.get_velocity().x, 
                                   vehicle.get_velocity().y])])
        u0 = reshape(u0, N, 2)
        u0[0:N-1,:] = u[1:N,:]
        u0[N-1,:]=u[N-1,:]

    time.sleep(10)

finally:
	for actor in actor_list:
		actor.destroy()
	print("All cleaned up!")