# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:42:27 2020

@author: vxr131730
"""


import pygame
import math


def draw_ellipse(A, B, width, color, line):
    """
    draws ellipse between two points
    A = start point (x,y)
    B = end point (x,y)
    width in pixel
    color (r,g,b)
    line thickness int, if line=0 fill ellipse
    """
    # point coordinates
    xA, yA = A[0], A[1]
    xB, yB = B[0], B[1]
    # calculate ellipse height, distance between A and B
    AB = math.sqrt((xB - xA)**2 + (yB - yA)**2)

    # difference between corner point coord and ellipse endpoint
    def sp(theta):
        return abs((width / 2 * math.sin(math.radians(theta))))

    def cp(theta):
        return abs((width / 2 * math.cos(math.radians(theta))))

    if xB >= xA and yB < yA:
        # NE quadrant
        theta = math.degrees(math.asin((yA - yB) / AB))
        xP = int(xA - sp(theta))
        yP = int(yB - cp(theta))
    elif xB < xA and yB <= yA:
        # NW
        theta = math.degrees(math.asin((yB - yA) / AB))
        xP = int(xB - sp(theta))
        yP = int(yB - cp(theta))
    elif xB <= xA and yB > yA:
        # SW
        theta = math.degrees(math.asin((yB - yA) / AB))
        xP = int(xB - sp(theta))
        yP = int(yA - cp(theta))
    else:
        # SE
        theta = math.degrees(math.asin((yA - yB) / AB))
        xP = int(xA - sp(theta))
        yP = int(yA - cp(theta))

    # create surface for ellipse
    ellipse_surface = pygame.Surface((AB, width), pygame.SRCALPHA)
    # draw surface onto ellipse
    pygame.draw.ellipse(ellipse_surface, color, (0, 0, AB, width), line)
    # rotate ellipse
    ellipse = pygame.transform.rotate(ellipse_surface, theta)
    # blit ellipse onto screen
    screen.blit(ellipse, (xP, yP))


screen = pygame.display.set_mode((1000, 1000))

running = True
while running:
    screen.fill((255, 250, 200))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    draw_ellipse((500, 500), (420, 350), 100, (0, 255, 0), 5)
    draw_ellipse((400, 600), (700, 280), 80, (255, 0, 0), 5)
    draw_ellipse((260, 190), (670, 440), 50, (0, 0, 255), 5)

    pygame.display.update()