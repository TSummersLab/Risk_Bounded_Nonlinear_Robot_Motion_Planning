"""Extra math functions"""
# Author: Ben Gravell
# Also see
# https://www.iquilezles.org/www/articles/functions/functions.htm
# https://thebookofshaders.com/05/

import numpy as np


def quadratic_formula(a, b, c):
    """Solve the quadratic equation 0 = a*x**2 + b*x + c using the quadratic formula."""
    if a == 0:
        return [-c/b, np.nan]
    disc = b**2 - 4*a*c
    disc_sqrt = disc**0.5
    den = 2*a
    roots = [(-b+disc_sqrt)/den, (-b-disc_sqrt)/den]
    return roots


def symlog(X,scale=1):
    """Symmetric log transform"""
    return np.multiply(np.sign(X),np.log(1+np.abs(X)/(10**scale)))


def mix(x, y, ratio):
    """Convex combination of two numbers"""
    return (1 - ratio)*x + ratio*y


def smoothstep(x):
    """Polynomial transition from 0 to 1 with continuous first derivative"""
    return -2*x**3 + 3*x**2


def bump(x, radius, width):
    """Offset, scaled, and mirrored smoothstep segments"""
    def f(x):
        return np.where(np.abs(x) < 1, np.where(x < 0, smoothstep(1+x), smoothstep(1-x)), 0)
    return f((x-radius)/(width/2))


def dwell_profile(angle, squash=20, narrow=0.95, phase=0):
    """Rise-dwell-fall-dwell profile"""
    def g(angle):
        return np.tanh(squash*np.sin(angle + np.pi*(narrow - phase)))
    return g(angle) + g(angle - np.pi*narrow)

def contrast(x, amount=0, tolerance=1e-9):
    '''
    Nonlinear contrast (or compression)
    x: Input data, should be between 0 and 1
    amount: Amount of contrast or compression
            Positive values are contrast, with amount=np.inf giving a step function
            Negative values are compression, with amount=-np.inf giving a constant function
            Zero value gives a linear function with unity slope
    '''
    if amount == 0:
        return x
    if np.isinf(amount):
        if amount > 0:
            return x > 0.5
        else:
            return np.ones_like(x)*0.5
    f = lambda y: 1/(1+np.exp(-np.abs(amount)*(y-0.5)))
    finv = lambda z: 0.5 - (1/np.abs(amount))*np.log(1/z - 1)
    f0 = f(0)
    if amount > 0:
        return (f(x)-f0)/(1-2*f0)
    else:
        z = f0+(1-2*f0)*x
        z = np.minimum(z, 1-tolerance)
        z = np.maximum(z, tolerance)
        return finv(z)