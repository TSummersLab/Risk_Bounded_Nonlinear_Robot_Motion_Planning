"""Functions for control of LTI systems."""
# Author: Ben Gravell

import numpy as np
from matrixmath import dlyap, mdot


def ctrb(A, B):
    """Controllabilty matrix
    Parameters
    ----------
    A, B: np.array
        Dynamics and input matrix of the system
    Returns
    -------
    C: matrix
        Controllability matrix
    Examples
    --------
    >>> C = ctrb(A, B)
    """

    n, m = np.shape(B)
    AiB = np.zeros([n,m,n])
    AiB[:,:,0] = np.copy(B)
    for i in range(n-1):
        AiB[:,:,i+1] = np.dot(A, AiB[:,:,i])
    ctrb_matrix = np.reshape(AiB, [n,n*m], order='F')
    return ctrb_matrix


def obsv(A, C):
    """Observability matrix
    Parameters
    ----------
    A, C: np.array
        Dynamics and output matrix of the system
    Returns
    -------
    O: matrix
        Observability matrix
    Examples
    --------
    >>> O = obsv(A, C)
   """

    return ctrb(A.T,  C.T).T


def dgram_ctrb(A, B):
    """Discrete-time controllability Gramian."""
    return dlyap(A, B.dot(B.T))


def dgram_obsv(A, C):
    """Discrete-time observability Gramian."""
    # return dlyap(A.T,C.T.dot(C))
    return dgram_ctrb(A.T, C.T)


def dctg(A, Q, t):
    """Discrete-time finite-horizon cost-to-go matrix"""
    P = np.copy(Q)
    Pt = np.copy(Q)
    for _ in range(t):
        Pt = mdot(A.T, Pt, A)
        P += Pt
    return P