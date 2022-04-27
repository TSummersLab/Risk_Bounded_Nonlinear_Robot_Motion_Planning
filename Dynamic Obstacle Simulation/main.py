#!/usr/bin/env python

"""
@author: vxr131730 - Venkatraman Renganathan

This code simulates the bicycle dynamics of car by steering it on the road by
avoiding another static car obstacle. The ego_vehicle has to consider all the
system and perception uncertainties to generate a risk-bounded motion plan and 
execute it with coherent risk assessment. This code uses the CARLA simulator.

CARLA SIMULATOR VERSION - 0.9.10
PYTHON VERSION          - 3.7.6
VISUAL STUDIO VERSION   - 2017
UNREAL ENGINE VERSION   - 4.24.3

This script is tested in Python 3.7.6, Windows 10, 64-bit Processor
(C) Venkatraman Renganathan, 2021.  Email: vrengana@utdallas.edu

This program is a free software: you can redistribute it and/or modify it
under the terms of the GNU lesser General Public License, either version 
3.7, or any later version. This program is distributed in the hope that it 
will be useful, but WITHOUT ANY WARRANTY. 

"""
###############################################################################
####################### Import all the required libraries #####################
###############################################################################

import glob
import math
import os
import sys
import random
import numpy as np
import cv2
import re
import weakref
import argparse
import collections
import datetime
import pickle
import itertools
import pygame
import config
import MPC_Tracker as MPC_Tracker

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

# =============================================================================
# -- Global Variables ---------------------------------------------------------
# =============================================================================

red                 = carla.Color(255, 0, 0)
green               = carla.Color(0, 255, 0)
blue                = carla.Color(47, 210, 231)
cyan                = carla.Color(0, 255, 255)
yellow              = carla.Color(255, 255, 0)
orange              = carla.Color(255, 162, 0)
white               = carla.Color(255, 255, 255)
plot_COLOR          = (248, 64, 24)
ellipse_color       = (0,255,0)
ellipse_life_time   = 1/100
path_life_time      = 30
trail_life_time     = 5
waypoint_separation = 5
main_display_width  = 1280
main_display_height = 720
mini_display_width  = int(1280/3)
mini_display_height = int(720/3)
main_display_fov    = 90
mini_display_fov    = 90
line_thickness      = 3


# =============================================================================
# -- Global functions ---------------------------------------------------------
# =============================================================================

def Wrap_Angle(angle_in_degree):
    angle_in_rad = angle_in_degree / 180.0 * np.pi
    while angle_in_rad > np.pi:
        angle_in_rad -= 2*np.pi
    while angle_in_rad <= -np.pi:
        angle_in_rad += 2*np.pi
    return angle_in_rad

###############################################################################
###############################################################################

def Find_Weather_Presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

###############################################################################
###############################################################################

def Get_Actor_Display_Name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

###############################################################################
###############################################################################

def Get_Vehicle_Wheelbases(wheels, center_of_mass):
    front_left_wheel = wheels[0]
    front_right_wheel = wheels[1]
    back_left_wheel = wheels[2]
    back_right_wheel = wheels[3]
    front_x = (front_left_wheel.position.x + front_right_wheel.position.x) / 2.0
    front_y = (front_left_wheel.position.y + front_right_wheel.position.y) / 2.0
    front_z = (front_left_wheel.position.z + front_right_wheel.position.z) / 2.0
    back_x = (back_left_wheel.position.x + back_right_wheel.position.x) / 2.0
    back_y = (back_left_wheel.position.y + back_right_wheel.position.y) / 2.0
    back_z = (back_left_wheel.position.z + back_right_wheel.position.z) / 2.0
    l = np.sqrt( (front_x - back_x)**2 + (front_y - back_y)**2 + (front_z - back_z)**2  ) / 100.0
    # print(f"center of mass : {center_of_mass.x}, {center_of_mass.y}, {center_of_mass.z} wheelbase {l}")
    # return center_of_mass.x , l - center_of_mass.x, l
    return l - center_of_mass.x, center_of_mass.x, l

# =============================================================================
# -- Plotting functionalities -------------------------------------------------
# =============================================================================
   
def Draw_Transform(debug, trans, col=carla.Color(255, 0, 0), lt=10):
    debug.draw_arrow(trans.location, trans.location + trans.get_forward_vector(), 
                 thickness=0.05, arrow_size=0.1, color=col, life_time=lt)

###########################################################################
###########################################################################

def Draw_Waypoint_Union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=10):
    debug.draw_line(w0.transform.location + carla.Location(z=0.25), 
                    w1.transform.location + carla.Location(z=0.25), 
                    thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 
                     0.1, color, lt, False)
        
###############################################################################
###############################################################################
    
def plotParametricEllipse(track_flag, car_world, x_ext, y_ext, center, 
                          orientation_angle = 0, 
                          color = carla.Color(r=0, g=0, b=0), 
                          nPoints = 20):        
        
    # a is semi major axis and b is semi minor axis
    a = np.sqrt(y_ext**2 + x_ext**2) - 1
    b = x_ext + 1
    
    # Generate data for ellipse structure
    t    = np.linspace(0,2*np.pi, nPoints)
    x    = (a)*np.cos(t)
    y    = (b)*np.sin(t)
    data = np.array([x,y])
    R    = np.array([[np.cos(orientation_angle), -np.sin(orientation_angle)],
                     [np.sin(orientation_angle), np.cos(orientation_angle)]])
    data = np.dot(R,data)
    
    # Center the ellipse at given center
    data[0] += center[0]
    data[1] += center[1]
    
    # Coordinates at each step
    tailCoord = carla.Location(x=0, y = 0, z = center[2])
    headCoord = carla.Location(x=0, y = 0, z = center[2])
    
    for i in range(len(data[0]) - 1):
        tailCoord.x = data[0][i]
        tailCoord.y = data[1][i]
        headCoord.x = data[0][i+1]
        headCoord.y = data[1][i+1]
        if track_flag:
            car_world.debug.draw_line(tailCoord, headCoord, 
                                      life_time=ellipse_life_time, 
                                      thickness=0.2, color = green)
        else:
            car_world.debug.draw_line(tailCoord, headCoord, 
                                      life_time=10, 
                                      thickness=0.2, color = green)

###############################################################################
###############################################################################            

def plotPath(car_world, plot_coords, refPathFlag):      
    
    for i in range(plot_coords.shape[1] - 1):        
        # Coordinates at each step
        tailCoord = carla.Location(x = plot_coords[0,i],
                                   y = plot_coords[1,i], 
                                   z = plot_coords[2,i])
        headCoord = carla.Location(x = plot_coords[0,i+1], 
                                   y = plot_coords[1,i+1], 
                                   z = plot_coords[2,i+1])
        if refPathFlag:      
            # Plot the reference trajectory using orange color
            car_world.debug.draw_line(tailCoord, headCoord, 
                                      life_time=60, thickness = 0.1, color = orange)
        else:
            # Plot the DR-RRT* Tree using green color
            car_world.debug.draw_line(tailCoord, headCoord, 
                                      life_time=60, thickness = 0.05, color = red)
                

###############################################################################
###############################################################################
            
# =============================================================================
# -- World --------------------------------------------------------------------
# =============================================================================

class World(object):
    def __init__(self, carla_world, display, hud, args):
        ## CREATE CLASS VARIABLES FROM THE ARGUMENTS        
        # Get the world, map and other simulation parameters
        self.world   = carla_world
        self.display = display
        self.map     = self.world.get_map()        
        self.debug   = self.world.debug
        self.hud     = hud
        self.args    = args
        # Get the actor information in the world
        
        self.player            = None
        self.static_player     = None
        self.goal_info         = None
        self._actor_filter     = args.filter
        self.actor_role        = args.rolename
        self.static_actor_role = args.rolename_static
        self.player_length     = 0.0
        self.player_width      = 0.0
        self.obstacle_length   = 0.0
        self.obstacle_width    = 0.0
        self.boundary_info     = [args.bnd_x, args.bnd_y, args.bnd_w, args.bnd_h]
        # Get the sensors in the world
        self.collision_sensor      = None
        self.lane_invasion_sensor  = None
        self.gnss_sensor           = None
        self.imu_sensor            = None
        self.radar_sensor          = None
        self.camera_manager        = None
        self._weather_presets      = Find_Weather_Presets()
        self._weather_index        = 0        
        self._gamma                = args.gamma
        self.player_max_speed      = 1.589
        self.player_max_speed_fast = 3.713
        # Restart the world with actors and sensors
        self.Restart()
        self.world.on_tick(hud.On_World_Tick)
        # Simulation recording parameters
        self.recording_enabled = False
        self.recording_start = 0             

    def Restart(self):        
        
        # Keep same camera config if the camera manager exists.
        cam_index     = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        minicam_index     = self.camera_manager.mini_index if self.camera_manager is not None else 0
        minicam_pos_index = self.camera_manager.mini_transform_index if self.camera_manager is not None else 0
        
        # Get the desired blueprint of the hero vehicle
        blueprints = self.world.get_blueprint_library().filter(self._actor_filter)
        
        # =====================================================================
        # ====================== SPAWN PLAYERS ================================
        # =====================================================================
        # PREPARE BLUEPRINT FOR HERO VEHICLE
        # Get the desired vehicle blueprint
        blueprint = None
        for blueprint_candidate in blueprints:            
            if blueprint_candidate.id == self.args.vehicle_id:
                blueprint = blueprint_candidate
                break
        # If not available, choose one in random
        if blueprint is None:
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        
        # Set the vehicle role as hero
        blueprint.set_attribute('role_name', self.actor_role)
        
        # Do some cosmetic changes on the vehicle blueprint
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
            
        # =====================================================================
        ## PREPARE BLUEPRINT FOR STATIC VEHICLE
        # Get the desired static vehicle blueprint
        static_blueprint  = None
        for blueprint_candidate in blueprints:            
            if blueprint_candidate.id == self.args.vehicle_id_static:
                static_blueprint = blueprint_candidate
                break
        # If not available, choose one in random
        if static_blueprint is None:
            static_blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        
        # Set the vehicle role as static
        static_blueprint.set_attribute('role_name', self.static_actor_role)
        
        # Do some cosmetic changes on the static vehicle blueprint
        if static_blueprint.has_attribute('color'):
            color = random.choice(static_blueprint.get_attribute('color').recommended_values)
            static_blueprint.set_attribute('color', color)
        if static_blueprint.has_attribute('driver_id'):
            driver_id = random.choice(static_blueprint.get_attribute('driver_id').recommended_values)
            static_blueprint.set_attribute('driver_id', driver_id)
        if static_blueprint.has_attribute('is_invincible'):
            static_blueprint.set_attribute('is_invincible', 'true')        
        
        # =====================================================================
        # Spawn the hero player using the vehicle blueprint
        if self.player is not None:
            spawn_point                = self.player.get_transform()
            spawn_point.location.z    += 2.0
            spawn_point.rotation.roll  = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        # If the player doesn't exist yet, try to spawn it as new player
        while self.player is None:            
            # Set a desired location to spawn the vehicle
            spawn_location               = carla.Location()
            spawn_location.x             = float(self.args.x)
            spawn_location.y             = float(self.args.y)
            spawn_location.z             = float(self.args.z)
            spawn_waypoint               = self.map.get_waypoint(spawn_location)
            spawn_transform              = spawn_waypoint.transform
            spawn_transform.location.z   = 1.0
            spawn_transform.rotation.yaw = float(self.args.yaw)
            
            # Try to spawn the vehicle with a desired blueprint & spawn_transform
            self.player = self.world.try_spawn_actor(blueprint, spawn_transform)
        
        # Check if successfully spawned the static player
        if self.player is not None:            
            # Infer the size of the static police obstacle car
            player_dim         = self.player.bounding_box.extent
            self.player_length = player_dim.x
            self.player_width  = player_dim.y  
            print('Length of car:', round(self.player_length,2),
                  'Width of car:', round(self.player_width,2))
        
        # =====================================================================
        # Spawn the static player using the static vehicle blueprint
        if self.static_player is not None:
            static_spawn_point                = self.static_player.get_transform()
            static_spawn_point.location.z    += 2.0
            static_spawn_point.rotation.roll  = 0.0
            static_spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.static_player = self.world.try_spawn_actor(static_blueprint, static_spawn_point)
            
        # If the player doesn't exist yet, try to spawn it as new player
        while self.static_player is None:            
            # Set a desired location to spawn the vehicle
            static_spawn_location               = carla.Location()
            static_spawn_location.x             = float(self.args.x_static)
            static_spawn_location.y             = float(self.args.y_static)
            static_spawn_location.z             = float(self.args.z_static)
            static_spawn_waypoint               = self.map.get_waypoint(static_spawn_location)
            static_spawn_transform              = static_spawn_waypoint.transform
            static_spawn_transform.location.z   = 1.0
            static_spawn_transform.rotation.yaw = float(self.args.yaw_static)
            
            # Try to spawn the vehicle with a desired blueprint & static_spawn_transform
            self.static_player = self.world.try_spawn_actor(static_blueprint, static_spawn_transform)
            
            # Set the handbrake = True so that police car is static
            self.static_player.apply_control(carla.VehicleControl(throttle=0.0, 
                                                                  steer=0.0, 
                                                                  hand_brake=True))
            
            # Check if successfully spawned the static player
            if self.static_player is not None:                             
                # Infer the size of the static police obstacle car
                static_dim           = self.static_player.bounding_box.extent
                self.obstacle_length = static_dim.x
                self.obstacle_width  = static_dim.y
                print('Obstacle Length:', round(static_dim.x,2), 
                      'Obstacle Width:', round(static_dim.y,2))                
        
        # =====================================================================
        # ============== SET THE GOAL REGION FOR THIS CARLA SIMULATION ========
        # =====================================================================       
        # Set the Goal point at a desired location with same orientation as start postion
        
        goal_location               = carla.Location()
        goal_location.x             = float(self.args.x_ped)
        goal_location.y             = float(self.args.y_ped)
        goal_location.z             = float(self.args.z_ped)
        goal_waypoint               = self.map.get_waypoint(goal_location)
        goal_transform              = goal_waypoint.transform
        goal_transform.location.z   = 1.0
        goal_transform.rotation.yaw = float(self.args.yaw)        
        
        # Update the goal_info variable
        self.goal_info        = {}
        self.goal_info["x"]   = goal_location.x
        self.goal_info["y"]   = goal_location.y
        self.goal_info["yaw"] = goal_transform.rotation.yaw
        self.goal_info["v"]   = 0.0                      
        
        # =====================================================================
        # ============== SET THE CONTROLLER FOR THIS CARLA SIMULATION =========
        # =====================================================================
        # Set the hero vehicle values to the controller
        physics_controls = self.player.get_physics_control()
        lf, lr, L = Get_Vehicle_Wheelbases(physics_controls.wheels, physics_controls.center_of_mass)          
        
        # Get obstacle information
        obs_w = [static_spawn_transform.location.x,
                 static_spawn_transform.location.y,
                 Wrap_Angle(static_spawn_transform.rotation.yaw),
                 self.obstacle_length,
                 self.obstacle_width]
        
        # Set the controller as the MPC Controller - Any other controller can be plugged in here 
        # Say, a learning based controller or LQR etc can be plugged in here
        # Set the controller Parameter settings as a dictionary
        controller_params                 = {}        
        controller_params["wheelbase"]    = L   # L = 2.9 from Carlasimulator      
        controller_params["obstacle"]     = obs_w          
        controller_params["goal_x_pos"]   = goal_location.x
        controller_params["goal_y_pos"]   = goal_location.y        
        controller_params["boundary"]     = self.boundary_info # bottom_left [x,y, width, height]
        
        # Set the controller as MPC Controller with controller_params
        # Get an object of the controller and store it as self.controller
        self.controller = MPC_Tracker.Controller(controller_params)
        
        # Infer the player parameters to update the controller
        velocity_vec      = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location  = current_transform.location
        current_rotation  = current_transform.rotation
        current_x         = current_location.x
        current_y         = current_location.y
        current_yaw       = current_rotation.yaw
        current_speed     = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        robot_state       = [current_x, current_y, current_yaw, current_speed]
        
        # Get the simulation information from the HUD sensor
        frame, current_timestamp = self.hud.Get_Simulation_Information()
        
        # Update the controller with current states and simulation information
        self.controller.Update_Information(robot_state, [obs_w[0], obs_w[1]],
                                           current_timestamp, frame)
        
        # =====================================================================
        # ============== SET UP ALL THE SENSORS WITH ACTORS IN WORLD ==========
        # =====================================================================        
        # Set up the sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.mini_transform_index = minicam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.set_mini_sensor(minicam_index, notify=False)
        actor_type = Get_Actor_Display_Name(self.player)
        self.hud.notification(actor_type)              
        
        # =====================================================================
        # =========== COMPUTE A MOTION PLAN FROM SOURCE TO DESTINATION ========
        # =====================================================================                
        # Call the motion planner to get the path from source to goal        
        # Load the pickle file which has the motion plan to get from source to goal        
        filename = 'waypts_ref_data.pkl'
        infile = open(filename,'rb')
        path_points = pickle.load(infile)
        infile.close()        
        self.ref_traj_waypts = path_points
        self.num_way_points = self.ref_traj_waypts.shape[1]
        print('Total Waypoints to track:', self.num_way_points)        
        
        # Get Ego Player Location
        start_loc = self.player.get_transform().location
        start_loc_z = start_loc.z        
        
        # Prepare a vector of length self.num_way_points with value start_loc_z
        start_z_vec = start_loc_z*np.ones((1,self.num_way_points))
        
        # Store the sequence of reference points - num_waypts x 3
        plot_points = np.concatenate((path_points[0:2,:], start_z_vec), axis=0)
        
        # Plot the goal region
        self.world.debug.draw_point(goal_location, size=0.3, 
                                    color=green, life_time=50.0)  
        
        # Plot the reference path
        print('Plotting Reference Trajectory')        
        plotPath(self.world, plot_points, refPathFlag = True)  
        
        # Plot the DR-RRT* Tree
        print('Plotting DR-RRT* Tree')        
        # Unbox Pickle file to load the nodeList data         
        infile = open('DR_RRT_Star_Tree.pkl','rb')
        nodeList = pickle.load(infile)
        infile.close()
        for ellipseNode in nodeList:
            if ellipseNode is not None and ellipseNode.parent is not None:                
                ellNodeShape = ellipseNode.means.shape  
                xPlotValues  = []
                yPlotValues  = []            
                # Prepare the trajectory x and y vectors and plot them                
                for k in range(ellNodeShape[0]):                                    
                    xPlotValues.append(ellipseNode.means[k,0,0])
                    yPlotValues.append(ellipseNode.means[k,1,0]) 
                # Plotting the risk bounded trajectories
                tree_z_data = start_loc_z*np.ones((1,ellNodeShape[0]))
                tree_xy_data = np.array((xPlotValues, yPlotValues))
                tree_xyz_data = np.concatenate((tree_xy_data, tree_z_data), axis=0)                
                plotPath(self.world, tree_xyz_data, refPathFlag = False)          
        
    ###########################################################################
    ###########################################################################

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset               = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.mini_sensor.destroy()
        self.camera_manager.sensor      = None
        self.camera_manager.mini_sensor = None
        self.camera_manager.index       = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [self.camera_manager.sensor, 
                   self.camera_manager.mini_sensor, 
                   self.collision_sensor.sensor, 
                   self.lane_invasion_sensor.sensor, 
                   self.gnss_sensor.sensor, 
                   self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy() 
        if self.static_player is not None:
            self.static_player.destroy() 

# =============================================================================
# -- HUD ----------------------------------------------------------------------
# =============================================================================


class HUD(object):
    def __init__(self, width, height, mini_width, mini_height):
        self.dim = (width, height, mini_width, mini_height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def On_World_Tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
        
    def Get_Simulation_Information(self):
        return self.frame, self.simulation_time
        
    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % Get_Actor_Display_Name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accel: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyro: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = Get_Actor_Display_Name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            display.blit(info_surface, (main_display_width - mini_display_width, 
                                        main_display_height - mini_display_height))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# =============================================================================
# -- FadingText ---------------------------------------------------------------
# =============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# =============================================================================
# -- HelpText -----------------------------------------------------------------
# =============================================================================

class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# =============================================================================
# -- CollisionSensor ----------------------------------------------------------
# =============================================================================

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = Get_Actor_Display_Name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# =============================================================================
# -- CameraManager ------------------------------------------------------------
# =============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.mini_sensor = None
        self.surface = None
        self.mini_surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-7.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=-8.0, y=-6, z=6.0), carla.Rotation(pitch=10.0, yaw=-5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=0.0, z=10.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]        
        world = self._parent.get_world()
        self.bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = self.bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)            
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)
            item.append(bp)
        self.index = None
        self.mini_index = None
    

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            # If the self.sensor is None, spawn the actor
            cam_transform_index = self.transform_index+3
            
            bp = self.bp_library.find(self.sensors[index][0])            
            bp.set_attribute('image_size_x', str(self.hud.dim[0]))
            bp.set_attribute('image_size_y', str(self.hud.dim[1])) 
            bp.set_attribute('fov', str(main_display_fov))             
            
            self.sensor = self._parent.get_world().spawn_actor(bp, 
                                                               self._camera_transforms[cam_transform_index][0], 
                                                               attach_to=self._parent, 
                                                               attachment_type=self._camera_transforms[cam_transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            mini_flag = 0
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, mini_flag))
            sensor_calibration = np.identity(3)
            sensor_calibration[0, 2] = main_display_width / 2.0
            sensor_calibration[1, 2] = main_display_height / 2.0
            sensor_calibration[0, 0] = sensor_calibration[1, 1] = main_display_width / (2.0 * np.tan(main_display_fov * np.pi / 360.0))
            self.sensor.calibration  = sensor_calibration
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index
        
    def set_mini_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)        
        needs_respawn = True if self.mini_index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.mini_index][2]))
        if needs_respawn:
            if self.mini_sensor is not None:
                self.mini_sensor.destroy()
                self.mini_surface = None     
            # If the self.mini_sensor is None, spawn the actor
            cam_transform_index = self.transform_index+5
            mini_cam_transform  = self._camera_transforms[cam_transform_index][0]  
            mini_cam_attacher   = self._camera_transforms[cam_transform_index][1]                                
            
            # Modify the blueprints of the mini_sensor
            bp = self.bp_library.find(self.sensors[index][0])            
            bp.set_attribute('image_size_x', str(self.hud.dim[2]))
            bp.set_attribute('image_size_y', str(self.hud.dim[3])) 
            bp.set_attribute('fov', str(mini_display_fov)) 
            
            # Spawn the actor
            self.mini_sensor = self._parent.get_world().spawn_actor(bp, mini_cam_transform, attach_to=self._parent, 
                                                                    attachment_type=mini_cam_attacher)
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            mini_flag = 1
            self.mini_sensor.listen(lambda image_mini: CameraManager._parse_image(weak_self, image_mini, mini_flag))
            mini_calibration = np.identity(3)
            mini_calibration[0, 2] = mini_display_width / 2.0
            mini_calibration[1, 2] = mini_display_height / 2.0
            mini_calibration[0, 0] = mini_calibration[1, 1] = mini_display_width / (2.0 * np.tan(mini_display_fov * np.pi / 360.0))
            self.mini_sensor.calibration = mini_calibration
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.mini_index = index
        
    def toggle_camera(self): # NOT USED
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)
        self.set_mini_sensor(self.mini_index, notify=False, force_respawn=True)

    def next_sensor(self): # NOT USED
        self.set_sensor(self.index + 1)
        self.set_mini_sensor(self.index + 1)

    def toggle_recording(self): # NOT USED
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None and self.mini_surface is not None:
            display.blit(self.surface, (0, 0))
            display.blit(self.mini_surface, (main_display_width - mini_display_width, 
                                             main_display_height - mini_display_height))

    @staticmethod
    def _parse_image(weak_self, image, mini_flag):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'): 
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if mini_flag == 1:
                self.mini_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            if mini_flag == 0:
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)
            
# =============================================================================
# -- VehicleControl -----------------------------------------------------------
# =============================================================================

class VehicleControl(object):
    def __init__(self, world, car_dim):                                                    
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.car_dim = car_dim
        
    # Plots ellipse with given center - Assumption: Ellipses in X-Y plane (Z is assumed constant)
    def plotEllipse(self, center, orientation, nPoints = 20):
        
        # a is semi major axis abd b is semi minor axis
        a = self.car_dim[1] + 0.5
        b = self.car_dim[0] + 0.5
        print('a=', a, 'b=',b)        
        
        # Center
        h = center.x
        k = center.y
        
        # Coordinates at each step
        tailCoord = carla.Location(x=0, y = 0, z = center.z)
        headCoord = carla.Location(x=0,y = 0, z = center.z)
               
        # step is the difference in x value between each point plotted
        step = (2*a)/nPoints
        
        # Upper ellipse
        for i in range(nPoints):            
            # Start position of the line
            tailCoord.x = h - a + i*step            
            tailCoord.y = k + np.sqrt(b*b*(((tailCoord.x - h)**2)/(a**2)))                        
            # End position of the line
            headCoord.x = h - a + (i+1)*step
            headCoord.y = k + np.sqrt(b*b*(((headCoord.x-h)**2)/(a**2)))            
            self.world.debug.draw_line(tailCoord, headCoord, life_time=ellipse_life_time, 
                                       thickness=0.2, color=green)                    
        # Lower ellipse
        for i in range(nPoints):
            # Start position of the line
            tailCoord.x = h - a + i*step
            tailCoord.y = k - np.sqrt(b*b*(1-((tailCoord.x - h)**2)/(a**2)))            
            # End position of the lineamsm
            headCoord.x = h - a + (i+1)*step
            headCoord.y = k - np.sqrt(b*b*(1-((headCoord.x - h)**2)/(a**2)))            
            self.world.debug.draw_line(tailCoord, headCoord, life_time=ellipse_life_time,                                      
                                       thickness=0.2, color=green)

    def Parse_Events(self, client, world, clock, ref_iterator): 
        
        # Infer the current location and orientation of hero vehicle         
        velocity_vec      = world.player.get_velocity()
        current_transform = world.player.get_transform()        
        current_x         = current_transform.location.x
        current_y         = current_transform.location.y
        current_yaw       = current_transform.rotation.yaw
        current_speed     = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        robot_state       = [current_x, current_y, current_yaw, current_speed]        
        
        # Infer the current location and orientation of static vehicle obstacle
        st_trans = world.static_player.get_transform()                
        static_x = st_trans.location.x
        static_y = st_trans.location.y
        ob_state = [static_x, static_y]
        
        # Get the current frame and timestamp from the HUD sensor
        frame, current_timestamp = world.hud.Get_Simulation_Information()
        ready_to_go = world.controller.Update_Information(robot_state, ob_state, current_timestamp, frame)        
        
        # Make the police vehicle to be static - apply handbrake
        world.static_player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, hand_brake=True))
        
        # If ready_to_go, then compute a new control & apply to the vehicle
        start_location = world.player.get_location()    
        
        # Get the reference trajectory
        ref_traj_waypts = world.ref_traj_waypts

        if ready_to_go:  
            
            # Get reference trajectory for next N time steps (N: planning horizon for MPC)
            horizon_ref_states = ref_traj_waypts[:, ref_iterator:ref_iterator+config.carTrackHorizon+1]   # N = 10             
            
            # Update the waypoint information to the controller
            world.controller.Update_Waypoint(horizon_ref_states)            
            
            # Compute the control inputs
            world.controller.Track_Waypoint(world.player, world.static_player, ref_iterator) 
            
            # Dummy Control for taking pics
            # world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))  
                                                                       
            # Get the new location 
            new_location = world.player.get_location()   
            # Get the new transform
            new_transform = world.player.get_transform()                 
            # Plot the line between successive locations of the ego_vehicle
            world.world.debug.draw_line(start_location, new_location, life_time=10, 
                                        thickness=0.2, color=carla.Color(r=0, g=0, b=255))            
            # Plot car center ellipse
            # yaw = np.radians(new_transform.rotation.yaw - 90)
            # center = np.array([new_location.x, new_location.y, new_location.z])
            # plotParametricEllipse(True, world,  
            #                       world.player_length, world.player_width, 
            #                       center, yaw, carla.Color(r=0, g=255, b=100))
             
            # Update the start location
            start_location = new_location 
            
                
# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def Get_Bounding_Boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.Get_Bounding_Box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def Draw_Bounding_Boxes(display, bounding_boxes, VIEW_WIDTH, VIEW_HEIGHT):
        """
        Draws bounding boxes on pygame display.
        """
        BB_COLOR = [(248, 64, 24),
                    (100, 200, 128)]
        
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for k, bbox in enumerate(bounding_boxes):
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            pygame.draw.line(bb_surface, BB_COLOR[k], points[0], points[1], line_thickness)            
            pygame.draw.line(bb_surface, BB_COLOR[k], points[1], points[2], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[2], points[3], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[3], points[0], line_thickness)
            # top
            pygame.draw.line(bb_surface, BB_COLOR[k], points[4], points[5], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[5], points[6], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[6], points[7], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[7], points[4], line_thickness)
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR[k], points[0], points[4], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[1], points[5], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[2], points[6], line_thickness)
            pygame.draw.line(bb_surface, BB_COLOR[k], points[3], points[7], line_thickness)
        display.blit(bb_surface, (0, 0))
        
    @staticmethod
    def Get_Plot_Points(vehicle, camera, plot_cords):
        """
        Returns plotting points for a vehicle based on camera view.
        """        
        cords_x_y_z = ClientSideBoundingBoxes.Vehicle_To_Sensor(plot_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        plot_pts = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_plot = np.concatenate([plot_pts[:, 0] / plot_pts[:, 2], plot_pts[:, 1] / plot_pts[:, 2], plot_pts[:, 2]], axis=1)
        return camera_plot

    @staticmethod
    def Get_Bounding_Box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes.Create_Boundingbox_Coordinates(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes.Vehicle_To_Sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def Create_Boundingbox_Coordinates(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def Vehicle_To_Sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes.Vehicle_To_World(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes.World_To_Sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def Vehicle_To_World(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.Get_Matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.Get_Matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def World_To_Sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.Get_Matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def Get_Matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

###############################################################################
# =============================================================================
# -- Game_Loop() --------------------------------------------------------------
# =============================================================================
###############################################################################

def Game_Loop(args):
    
    # Give a name to the pygame display window
    pygame.init()
    pygame.display.set_caption('Risk Bounded Motion Planner')
    pygame.font.init()
    world = None

    try:        
        
        # First of all, we need to create the client that will send the request
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost(hosted in the same machine) at port 2000.
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.time_out)
        
        # Set the display
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        # Get the HUD Sensor
        hud = HUD(args.width, args.height, args.mini_width, args.mini_height)
        
        # Once we have a client, retrieve the world that is currently running.
        world = World(client.get_world(), display, hud, args)        

        # Call the controller and set up the problem  
        car_dim = [world.player.bounding_box.extent.x, world.player.bounding_box.extent.y]
        vehicleControl = VehicleControl(world, car_dim)
        
        # create a clock object to help track time
        clock = pygame.time.Clock()
        
        # Get all the actors of type vehicle
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        # Get the camera sensor object and define a calibration for it
        camera_obj             = world.camera_manager.sensor        
        calibration            = np.identity(3)
        calibration[0, 2]      = args.width / 2.0
        calibration[1, 2]      = args.height / 2.0
        calibration[0, 0]      = calibration[1, 1] = args.width / (2.0 * np.tan(args.view_fov * np.pi / 360.0))
        camera_obj.calibration = calibration                        
            
        # Define a flag to run the gameloop        
        run_game = True  
        print('Starting to track the Reference Trajectory!')
        
        # Initialize ref_iterator
        ref_iterator = 0
        
        # List to store car's position at each time
        car_x = []
        car_y = []        
        
        while run_game:                         
            
            # Capture events so that simulator updates fluently & stops smoothly
            # Very important - Else program stops after 2 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_game = False
            
            # Store car's position
            car_transform = world.player.get_transform()
            car_x.append(car_transform.location.x)
            car_y.append(car_transform.location.y)
            
            # Update the clock - Limit it to run at the desired Frames Per Sec
            clock.tick_busy_loop(args.FPS)   
            
            # Increment the reference iterator
            ref_iterator = ref_iterator + 1
            
            if vehicleControl.Parse_Events(client, world, clock, ref_iterator):
                return  
            
            # Tell the simulator to perform a tick - advance one step
            world.tick(clock)
            
            # Ask the world to render the display
            world.render(display)            
            
            # Update the bounding boxes of all the vehicles in the world using main camera
            bounding_boxes = ClientSideBoundingBoxes.Get_Bounding_Boxes(vehicles, camera_obj)
            ClientSideBoundingBoxes.Draw_Bounding_Boxes(display, bounding_boxes, args.width, args.height)
            
            # Update the contents of the entire display
            pygame.display.flip()                  

    finally:
        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.        
        print('Destroying Actors!')        
        if (world and world.recording_enabled):
            client.stop_recorder()
        if world is not None:
            world.destroy()
        print('Environment is Clean !')
        
        # Pickle dump the car tracking details
        print('Storing the tracking details')
        car_tracking_data = np.array((car_x, car_y))
        filename = 'car_tracking_data.pkl'
        outfile  = open(filename,'wb')
        pickle.dump(car_tracking_data, outfile)
        outfile.close() 
        
        # Stop the simulation
        pygame.display.quit()
        pygame.quit()
        sys.exit(0)
        
        
###############################################################################
###############################################################################

def main():
    
    # Define argparser
    argparser = argparse.ArgumentParser()
    
    # Add the arguments
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--vehicle_id', metavar='NAME', default='vehicle.audi.etron', help='actor filter (default: "vehicle.audi.etron")')
    argparser.add_argument('--vehicle_id_static', metavar='NAME', default='vehicle.tesla.model3', help='actor filter (default: "vehicle.tesla.model3☺")')
    argparser.add_argument('--rolename', metavar='NAME', default='hero', help='actor role name (default: "hero")')
    argparser.add_argument('--rolename_static', metavar='NAME', default='police', help='static actor role name (default: "police")')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument('--view_fov', default=90, type=float, help='Field of view of the camera (default: 90)')
    argparser.add_argument('--view_fov_mini', default=110, type=float, help='Field of view of the 2nd camera (default: 110)')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-i', '--info', action='store_true', help='Show text information')
    argparser.add_argument('-x', default=-75.0, type=float, help='X start position (default: 0.0)')
    argparser.add_argument('-y', default=-45.0, type=float, help='Y start position (default: 0.0)')
    argparser.add_argument('-z', default=0.0, type=float, help='Z start position (default: 0.0)')
    argparser.add_argument('-roll', default=0.0, type=float, help='Roll (default: 0.0)')
    argparser.add_argument('-pitch', default=0.0, type=float, help='Pitch (default: 0.0)')
    argparser.add_argument('-yaw', default=270, type=float, help='Yaw (default: 0.0)')
    argparser.add_argument('-x_static', default=-75.0, type=float, help='X start position (default: 0.0)')
    argparser.add_argument('-y_static', default=-65.0, type=float, help='Y start position (default: 0.0)')
    argparser.add_argument('-z_static', default=0.0, type=float, help='Z start position (default: 0.0)')
    argparser.add_argument('-roll_static', default=0.0, type=float, help='Roll (default: 0.0)')
    argparser.add_argument('-pitch_static', default=0.0, type=float, help='Pitch (default: 0.0)')
    argparser.add_argument('-yaw_static', default=270, type=float, help='Yaw (default: 0.0)')
    argparser.add_argument('-x_ped', default=-75.0, type=float, help='Pedestrian X position (default: 0.0)')
    argparser.add_argument('-y_ped', default=-75.0, type=float, help='Pedestrian Y position (default: 0.0)')
    argparser.add_argument('-z_ped', default=1.0, type=float, help='Pedestrian Z position (default: 0.0)')
    argparser.add_argument('-bnd_x', default=-83.0, type=float, help='Boundary X position (default: 0.0)')
    argparser.add_argument('-bnd_y', default=-25.0, type=float, help='Boundary Y position (default: 0.0)')
    argparser.add_argument('-bnd_w', default=13.0, type=float, help='Boundary Width (default: 0.0)')
    argparser.add_argument('-bnd_h', default=-77.0, type=float, help='Boundary Height (default: 0.0)')
    argparser.add_argument('-time_out', default=4.0, type=float, help='TimeOut (default: 10.0)')
    argparser.add_argument('-s', '--seed', metavar='S', default=os.getpid(), type=int, help='Seed for the random path (default: program pid)')
    argparser.add_argument('-t', '--tick-time', metavar='T', default=0.2, type=float, help='Tick time between updates (forward velocity) (default: 0.2)')
    argparser.add_argument('--FPS', metavar='FPS', default='60', type=int, help='Frame per second for simulation')
    
    # Define the args method
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')] 
    args.mini_width, args.mini_height = int(args.width/3), int(args.height/3)
    
    try:
        Game_Loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        

###############################################################################
###############################################################################

if __name__ == '__main__':

    main()

###############################################################################
###############################################################################