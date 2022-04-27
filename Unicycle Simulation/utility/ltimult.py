"""Functions for control of LTI systems with multiplicative noise."""
# Author: Ben Gravell

import numpy as np
from numpy import linalg as la
from matrixmath import is_pos_def, vec, sympart, kron, dlyap, mdot
from extramath import quadratic_formula

import warnings
from warnings import warn


def dlyap_mult(A,B,K,a,Aa,b,Bb,Q,R,S0,matrixtype='P',algo='iterative',show_warn=False,check_pd=False,P00=None,S00=None):
    """
    Solve a discrete-time generalized Lyapunov equation
    for stochastic linear systems with multiplicative noise.
    """

    n = A.shape[1]
    n2 = n*n
    p = len(a)
    q = len(b)
    AK = A + np.dot(B,K)
    stable = True
    if algo=='linsolve':
        if matrixtype=='P':
            # Intermediate terms
            Aunc_P = np.zeros([n2,n2])
            for i in range(p):
                Aunc_P = Aunc_P + a[i]*kron(Aa[i].T)
            BKunc_P = np.zeros([n2,n2])
            for j in range(q):
                BKunc_P = BKunc_P + b[j]*kron(np.dot(K.T,Bb[j].T))
            # Compute matrix and vector for the linear equation solver
            Alin_P = np.eye(n2) - kron(AK.T) - Aunc_P - BKunc_P
            blin_P = vec(Q) + np.dot(kron(K.T),vec(R))
            # Solve linear equations
            xlin_P = la.solve(Alin_P,blin_P)
            # Reshape
            P = np.reshape(xlin_P,[n,n])
            if check_pd:
                stable = is_pos_def(P)
        elif matrixtype=='S':
            # Intermediate terms
            Aunc_S = np.zeros([n2,n2])
            for i in range(p):
                Aunc_S = Aunc_S + a[i]*kron(Aa[i])
            BKunc_S = np.zeros([n2,n2])
            for j in range(q):
                BKunc_S = BKunc_S + b[j]*kron(np.dot(Bb[j],K))
            # Compute matrix and vector for the linear equation solver
            Alin_S = np.eye(n2) - kron(AK) - Aunc_S - BKunc_S
            blin_S = vec(S0)
            # Solve linear equations
            xlin_S = la.solve(Alin_S,blin_S)
            # Reshape
            S = np.reshape(xlin_S,[n,n])
            if check_pd:
                stable = is_pos_def(S)
        elif matrixtype=='PS':
            P = dlyap_mult(A,B,K,a,Aa,b,Bb,Q,R,S0,matrixtype='P',algo='linsolve')
            S = dlyap_mult(A,B,K,a,Aa,b,Bb,Q,R,S0,matrixtype='S',algo='linsolve')
    elif algo=='iterative':
        # Implicit iterative solution to generalized discrete Lyapunov equation
        # Inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7553367
        # In turn inspired by https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122100X0020X/1-s2.0-089812219500119J/main.pdf?x-amz-security-token=AgoJb3JpZ2luX2VjECgaCXVzLWVhc3QtMSJIMEYCIQD#2F00Re8b3wnBnFpZQrjkOeXrNI4bYZ1J6#2F9BcJptZYAAIhAOQjTsZX573uFFEr7QveHx4NaZYWxlZfRN6hr5h1GJWWKuMDCOD#2F#2F#2F#2F#2F#2F#2F#2F#2F#2FwEQAhoMMDU5MDAzNTQ2ODY1IgxqkGe6i8wGmEj6YAwqtwNDKbotYDExP2D6PO8MrlIKYmHCtJhTu1CXLv0N5NKsYT90H2rJTNU0MvqsUsnXtbn6C9t9ed31XTf#2BHc7KrGmpOils7zgrjV1QG4LP0Fu2OcT4#2F#2FOGLWNvVjWY9gOLEHSeG5LhvBbxJiZVrI#2Bm1QAIVz5dxH5DVB27A2e9OmRrswrpPWuxQV#2BUvLkz2dVM4qSkvaDA#2F3KEJk9s0XE74mjO4ZHX7d9Q2aYwxsvFbII6Hms#2FZmB6125tBTwzd0K5xDit5kaoiYadOetp3M#2FvCdaiO0QeQwkV4#2FUaprOIIQGwJaMJuMNe7xInQxF#2B#2FmER81JhWEpBHBmz#2F5p0d2tU7F2oTDc2OR#2BV5dTKab47zgUw648fDT7ays0TQzqTMGnGcX9wIQpxSCam2E8Bhg6tsEs0#2FudddgnsiId368q70xai6ucMfabMSCqnv7O0OZqPVwY5b7qk4mxKIehpIzV6rrtXSAGrH95WGlgGz#2Fhmg9Qq6AUtb8NSqyYw0uZ00E#2FPZmNTnI3nwxjOA5qhyEbw3uXogRwYrv0dLkd50s7oO3mlYFeJDBurhx11t9p94dFqQq7sDY70m#2F4xMNCcmuUFOrMBY1JZuqtQ7QFBVbgzV#2B4xSHV6#2FyD#2F4ezttczZY3eSASJpdC4rjYHXcliiE7KOBHivchFZMIYeF3J4Nvn6UykX5sNfRANC2BDPrgoCQUp95IE5kgYGB8iEISlp40ahVXK62GhEASJxMjJTI9cJ2M#2Ff#2BJkwmqAGjTsBwjxkgiLlHc63rBAEJ2e7xoTwDDql3FSSYcvKzwioLfet#2FvXWvjPzz44tB3#2BTvYamM0uq47XPlUFcTrw#3D&AWSAccessKeyId=ASIAQ3PHCVTYWXNG3EKG&Expires=1554423148&Signature=Ysi80usGGEjPCvw#2BENTSD90NgVs#3D&hash=e5cf30dad62b0b57d7b7f5ba524cccacdbb36d2f747746e7fbebb7717b415820&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=089812219500119J&tid=spdf-a9dae0e9-65fd-4f31-bf3f-e0952eb4176c&sid=5c8c88eb95ed9742632ae57532a4a6e1c6b1gxrqa&type=client
        # Faster for large systems i.e. >50 states
        # Options
        max_iters = 1000
        epsilon_P = 1e-5
        epsilon_S = 1e-5
        # Initialize
        if matrixtype=='P' or matrixtype=='PS':
            if P00 is None:
                P = np.zeros([n,n])
            else:
                P = P00
        if matrixtype=='S' or matrixtype=='PS':
            if S00 is None:
                S = np.zeros([n,n])
            else:
                S = S00
        iterc = 0
        converged = False
        stop = False
        while not stop:
            if matrixtype=='P' or matrixtype=='PS':
                P_prev = P
                APAunc = np.zeros([n,n])
                for i in range(p):
                    APAunc += a[i]*mdot(Aa[i].T,P,Aa[i])
                BPBunc = np.zeros([n,n])
                for j in range(q):
                    BPBunc += b[j]*mdot(K.T,Bb[j].T,P,Bb[j],K)
                AAP = AK.T
                QQP = sympart(Q + mdot(K.T,R,K) + APAunc + BPBunc)
                P = dlyap(AAP,QQP)
                if np.any(np.isnan(P)) or not is_pos_def(P):
                    stable = False
                converged_P = la.norm(P-P_prev,2)/la.norm(P,2) < epsilon_P
            if matrixtype=='S' or matrixtype=='PS':
                S_prev = S
                ASAunc = np.zeros([n,n])
                for i in range(p):
                    ASAunc += a[i]*mdot(Aa[i],S,Aa[i].T)
                BSBunc = np.zeros([n,n])
                for j in range(q):
                    BSBunc = b[j]*mdot(Bb[j],K,S,K.T,Bb[j].T)
                AAS = AK
                QQS = sympart(S0 + ASAunc + BSBunc)
                S = dlyap(AAS,QQS)
                if np.any(np.isnan(S)) or not is_pos_def(S):
                    stable = False
                converged_S = la.norm(S-S_prev,2)/la.norm(S,2) < epsilon_S
            # Check for stopping condition
            if matrixtype=='P':
                converged = converged_P
            elif matrixtype=='S':
                converged = converged_S
            elif matrixtype=='PS':
                converged = converged_P and converged_S
            if iterc >= max_iters:
                stable = False
            else:
                iterc += 1
            stop = converged or not stable
#        print('\ndlyap iters = %s' % str(iterc))

    elif algo=='finite_horizon':
        P = np.copy(Q)
        Pt = np.copy(Q)
        S = np.copy(Q)
        St = np.copy(Q)
        converged = False
        stop = False
        while not stop:
            if matrixtype=='P' or matrixtype=='PS':
                APAunc = np.zeros([n,n])
                for i in range(p):
                    APAunc += a[i]*mdot(Aa[i].T,Pt,Aa[i])
                BPBunc = np.zeros([n,n])
                for j in range(q):
                    BPBunc += b[j]*mdot(K.T,Bb[j].T,Pt,Bb[j],K)
                Pt = mdot(AK.T,Pt,AK)+APAunc+BPBunc
                P += Pt
                converged_P = np.abs(Pt).sum() < 1e-15
                stable = np.abs(P).sum() < 1e10
            if matrixtype=='S' or matrixtype=='PS':
                ASAunc = np.zeros([n,n])
                for i in range(p):
                    ASAunc += a[i]*mdot(Aa[i],St,Aa[i].T)
                BSBunc = np.zeros([n,n])
                for j in range(q):
                    BSBunc = b[j]*mdot(Bb[j],K,St,K.T,Bb[j].T)
                St = mdot(AK,Pt,AK.T)+ASAunc+BSBunc
                S += St
                converged_S = np.abs(St).sum() < 1e-15
                stable = np.abs(S).sum() < 1e10
            if matrixtype=='P':
                converged = converged_P
            elif matrixtype=='S':
                converged = converged_S
            elif matrixtype=='PS':
                converged = converged_P and converged_S
            stop = converged or not stable
    if not stable:
        P = None
        S = None
        if show_warn:
            warnings.simplefilter('always', UserWarning)
            warn('System is possibly not mean-square stable')
    if matrixtype=='P':
        return P
    elif matrixtype=='S':
        return S
    elif matrixtype=='PS':
        return P, S


def dare_mult(A, B, a, Aa, b, Bb, Q, R, algo='iterative', show_warn=False):
    """
    Solve a discrete-time generalized algebraic Riccati equation
    for stochastic linear systems with multiplicative noise.
    """

    n = A.shape[1]
    m = B.shape[1]
    p = len(a)
    q = len(b)

    failed = False

    # Handle the scalar case more efficiently by solving the ARE exactly
    if n == 1 and m == 1:
        if p == 1 and q == 1:
            algo = 'scalar'
        else:
            try:
                if np.count_nonzero(a[1:]) == 0 and np.count_nonzero(b[1:]) == 0:
                    algo = 'scalar'
                elif np.sum(np.count_nonzero(Aa[1:], axis=(1,2))) == 0 \
                 and np.sum(np.count_nonzero(Bb[1:], axis=(1,2))) == 0:
                    algo = 'scalar'
            except:
                pass

    if algo == 'scalar':
        A2 = A[0,0]**2
        B2 = B[0,0]**2
        Aa2 = Aa[0,0]**2
        Bb2 = Bb[0,0]**2

        aAa2 = a[0]*Aa2
        bBb2 = b[0]*Bb2

        aAa2m1 = aAa2-1
        B2pbBb2 = B2+bBb2

        aa = aAa2m1*B2pbBb2+A2*bBb2
        bb = R[0,0]*(A2+aAa2m1) + Q[0,0]*B2pbBb2
        cc = Q[0,0]*R[0,0]

        roots = np.array(quadratic_formula(aa, bb, cc))

        if not(roots[0] > 0 or roots[1] > 0):
            failed = True
        else:
            P = roots[roots > 0][0]*np.eye(1)
            K = -B*P*A / (R+B2pbBb2*P)

    elif algo=='iterative':
        # Options
        max_iters = 1000
        epsilon = 1e-6
        Pelmax = 1e40

        # Initialize
        P = Q
        counter = 0
        converged = False
        stop = False

        while not stop:
            # Record previous iterate
            P_prev = np.copy(P)
            # Certain part
            APAcer = mdot(A.T, P, A)
            BPBcer = mdot(B.T, P, B)
            # Uncertain part
            APAunc = np.zeros([n,n])
            for i in range(p):
                APAunc += a[i]*mdot(Aa[i].T, P, Aa[i])
            BPBunc = np.zeros([m,m])
            for j in range(q):
                BPBunc += b[j]*mdot(Bb[j].T, P, Bb[j])
            APAsum = APAcer+APAunc
            BPBsum = BPBcer+BPBunc
            # Recurse
            P = Q + APAsum - mdot(A.T, P, B, la.solve(R+BPBsum, B.T), P, A)

            # Check for stopping condition
            if la.norm(P-P_prev,'fro')/la.norm(P,'fro') < epsilon:
                converged = True
            if counter >= max_iters or np.any(np.abs(P) > Pelmax):
                failed = True
            else:
                counter += 1
            stop = converged or failed

        # Compute the gains
        if not failed:
            K = -mdot(la.solve(R+BPBsum, B.T), P, A)

        if np.any(np.isnan(P)):
            failed = True

    if failed:
        if show_warn:
            warnings.simplefilter('always', UserWarning)
            warn("Recursion failed, ensure system is mean square stabilizable "
                 "or increase maximum iterations")
        P = None
        K = None

    return P, K