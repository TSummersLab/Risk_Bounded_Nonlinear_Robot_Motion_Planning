# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:15:44 2020

@author: Aadi
"""
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH1 = 1920//2
VIEW_HEIGHT1 = 1080//2

VIEW_WIDTH2 = 1920//6
VIEW_HEIGHT2 = 1080//6

VIEW_FOV1 = 90
VIEW_FOV2 = 110 # Top view - smaller display

BB_COLOR = (248, 64, 24)

class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera1 = None
        self.camera2 = None
        self.car = None

        self.display1 = None
        self.display2 = None
        self.image1 = None
        self.capture1 = True
        
        self.image2 = None
        self.capture2 = True

    def camera_blueprint1(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH1))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT1))
        camera_bp.set_attribute('fov', str(VIEW_FOV1))
        return camera_bp

    def camera_blueprint2(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH2))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT2))
        camera_bp.set_attribute('fov', str(VIEW_FOV2))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera1(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera1 = self.world.spawn_actor(self.camera_blueprint1(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera1.listen(lambda image1: weak_self().set_image(weak_self, image1,1))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH1 / 2.0
        calibration[1, 2] = VIEW_HEIGHT1 / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH1 / (2.0 * np.tan(VIEW_FOV1 * np.pi / 360.0))
        self.camera1.calibration = calibration

    def setup_camera2(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=3, z=8), carla.Rotation(pitch=-90))
        self.camera2 = self.world.spawn_actor(self.camera_blueprint2(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera2.listen(lambda image2: weak_self().set_image(weak_self, image2,2))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH2 / 2.0
        calibration[1, 2] = VIEW_HEIGHT2 / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH2 / (2.0 * np.tan(VIEW_FOV2 * np.pi / 360.0))
        self.camera2.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img, camNo):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if camNo==1 and self.capture1:
            self.image1 = img
            self.capture1 = False
        if camNo==2 and self.capture2:
            self.image2 = img
            self.capture2 = False

    def render1(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image1 is not None:
            array = np.frombuffer(self.image1.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image1.height, self.image1.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            
    def render2(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image2 is not None:
            array = np.frombuffer(self.image2.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image2.height, self.image2.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (VIEW_WIDTH1 - VIEW_WIDTH2, VIEW_HEIGHT1 - VIEW_HEIGHT2))           

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera1()
            self.setup_camera2()
            
            self.display = pygame.display.set_mode((VIEW_WIDTH1, VIEW_HEIGHT1), pygame.HWSURFACE | pygame.DOUBLEBUF)

            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            while True:
                self.world.tick()

                self.capture1 = True
                self.capture2 = True
                pygame_clock.tick_busy_loop(20)

                self.render1(self.display)
                self.render2(self.display)
                # Testing plotting by draw.line
                # bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                # ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                #bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
                #print(vehicles[0])
                #loc1 = vehicles[0].bounding_box.location
                #loc = vehicles[0].get_location()
                #curLoc = (loc.x, loc.y)
                
                #pygame.draw.line(bb_surface, BB_COLOR, curLoc ,(loc.x + 100, loc.y + 100))                
                #pygame.draw.line(bb_surface, BB_COLOR, (1,1) ,(100,100))  
                
                # End testing
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera1.destroy()
            self.camera2.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
