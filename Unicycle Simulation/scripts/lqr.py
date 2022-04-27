#!/usr/bin/env python3
"""
Changelog:
New in version 1_0:
- Create script with lqr and lqrm functions

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author:
Ben Gravell
Email:
benjamin.gravell@utdallas.edu
Github:
@BenGravell

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LQR functions used as low level controllers (to track RRT* plan)
Supported variations:
- Generalized Linear Quadratic Regulator (lqr)
- LQR with multiplicative noise and additive adversary (lqrm)

Tested platform:
- TODO: add platform

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import sys
sys.path.insert(0, '../utility')
from utility.matrixmath import mdot, sympart


def lqr(z_hist, A_hist, B_hist, G_hist):
    """
    Solve a generalized linear-quadratic regulation problem
    minimize sum([x[k].T, u[k].T, z[k].T] @ G[k] @ [x[k].T, u[k].T, z[k].T].T)
    subject to x[k+1] = A[k] @ x[k] + B[k] @ u[k]
    where
    A[k], B[k] are dynamics matrices of size n x n and n x m respectively
    G[k] are positive semidefinite cost matrices of size n+m+p x n+m+p
    z[k] is a given exogenous signal
    """

    n, m, p = A_hist.shape[2], B_hist.shape[2], z_hist.shape[1]
    T = A_hist.shape[0]

    P_hist = np.zeros([T, n, n])
    q_hist = np.zeros([T, n])
    r_hist = np.zeros(T)

    K_hist = np.zeros([T, m, n])
    L_hist = np.zeros([T, m, p])
    e_hist = np.zeros([T, m])

    # Initialize cost
    z = z_hist[-1]
    G = G_hist[-1]
    Gxx = G[0:n, 0:n]
    Gxz = G[0:n, n+m:n+m+p]
    Gzz = G[n+m:n+m+p, n+m:n+m+p]
    P = Gxx
    q = np.dot(Gxz, z)
    r = mdot(z.T, Gzz, z)

    # Recurse
    for t in range(T-1, -1, -1):
        # Symmetrize for numerical stability
        Pqr = np.block([[P, q[:, None]],
                        [q[None, :], r]])
        Pqr = sympart(Pqr)
        P = Pqr[0:n, 0:n]
        q = Pqr[0:n, -1]
        r = Pqr[-1, -1]

        # Record cost function
        P_hist[t] = P
        q_hist[t] = q
        r_hist[t] = r

        # Extract dynamics, cost, and exogenous signal data
        z = z_hist[t]
        A = A_hist[t]
        B = B_hist[t]

        G = G_hist[t]
        Gxx = G[0:n, 0:n]
        Gxu = G[0:n, n:n+m]
        Gxz = G[0:n, n+m:n+m+p]
        Gux = G[n:n+m, 0:n]
        Guu = G[n:n+m, n:n+m]
        Guz = G[n:n+m, n+m:n+m+p]
        Gzx = G[n+m:n+m+p, 0:n]
        Gzu = G[n+m:n+m+p, n:n+m]
        Gzz = G[n+m:n+m+p, n+m:n+m+p]

        Hux = Gux + mdot(B.T, P, A)
        Huu = Guu + mdot(B.T, P, B)

        # Compute gains
        K = -la.solve(Huu, Hux)
        L = -la.solve(Huu, Guz)
        e = -la.solve(Huu, np.dot(B.T, q))

        # Record gains
        K_hist[t] = K
        L_hist[t] = L
        e_hist[t] = e

        # Compute intermediate quantities
        IK = np.vstack([np.eye(n), K])
        AK = A + np.dot(B, K)
        Gxxuu = np.block([[Gxx, Gxu],
                          [Gux, Guu]])
        Guxzu = np.block([[Gux, Guu],
                          [Gzx, Gzu]])
        Guuzz = np.block([[Guu, Guz],
                          [Gzu, Gzz]])
        s = np.dot(L, z) + e
        sz = np.hstack([s, z])
        Bs1 = np.hstack([np.dot(B, s), 1])

        # Update cost function
        P_new = mdot(IK.T, Gxxuu, IK) + mdot(AK.T, P, AK)
        q_new = mdot(IK.T, Guxzu.T, sz) + np.dot(AK.T, mdot(P, B, s) + q)
        r_new = mdot(sz.T, Guuzz, sz) + mdot(Bs1.T, Pqr, Bs1)
        P, q, r = P_new, q_new, r_new

    return K_hist, L_hist, e_hist, P_hist, q_hist, r_hist


def lqrm(z_hist, A_hist, B_hist, C_hist, Ai_hist, Bi_hist, Ci_hist, alpha_var_hist, beta_var_hist, gamma_var_hist,
         G_hist, E_hist, W_hist):
    """
    Solve a generalized linear-quadratic regulation problem with multiplicative noise and additive adversary
    """

    n, m, c, p = A_hist.shape[2], B_hist.shape[2], C_hist.shape[2], z_hist.shape[1]
    num_alphas = alpha_var_hist.shape[1]
    num_betas = beta_var_hist.shape[1]
    num_gammas = gamma_var_hist.shape[1]
    T = A_hist.shape[0]

    P_hist = np.zeros([T, n, n])
    q_hist = np.zeros([T, n])
    r_hist = np.zeros(T)

    Ku_hist = np.zeros([T, m, n])
    Lu_hist = np.zeros([T, m, p])
    eu_hist = np.zeros([T, m])
    Kv_hist = np.zeros([T, c, n])
    Lv_hist = np.zeros([T, c, p])
    ev_hist = np.zeros([T, c])

    # Initialize cost
    z = z_hist[-1]
    G = G_hist[-1]
    Gxx = G[0:n, 0:n]
    Gxz = G[0:n, n+m+c:n+m+c+p]
    Gzz = G[n+m+c:n+m+c+p, n+m+c:n+m+c+p]
    P = Gxx
    q = np.dot(Gxz, z)
    r = mdot(z.T, Gzz, z)

    # Recurse
    for t in range(T-1, -1, -1):
        # Symmetrize for numerical stability
        Pqr = np.block([[P, q[:, None]],
                        [q[None, :], r]])
        Pqr = sympart(Pqr)
        P = Pqr[0:n, 0:n]
        q = Pqr[0:n, -1]
        r = Pqr[-1, -1]

        # Record cost function
        P_hist[t] = P
        q_hist[t] = q
        r_hist[t] = r

        # Extract dynamics, cost, and exogenous signal data
        A = A_hist[t]
        B = B_hist[t]
        C = C_hist[t]
        Ai = Ai_hist[t]
        Bi = Bi_hist[t]
        Ci = Ci_hist[t]
        alpha_var = alpha_var_hist[t]
        beta_var = beta_var_hist[t]
        gamma_var = gamma_var_hist[t]

        G = G_hist[t]
        W = W_hist[t]
        E = E_hist[t]
        z = z_hist[t]

        ABC0 = np.hstack([A, B, C, np.zeros([n, p])])
        APA = np.zeros([n, n])
        for i in range(num_alphas):
            APA += alpha_var[i]*mdot(Ai[i].T, P, Ai[i])
        BPB = np.zeros([m, m])
        for i in range(num_betas):
            BPB += beta_var[i]*mdot(Bi[i].T, P, Bi[i])
        CPC = np.zeros([c, c])
        for i in range(num_gammas):
            CPC += gamma_var[i]*mdot(Ci[i].T, P, Ci[i])

        H = G + mdot(ABC0.T, P, ABC0) + sla.block_diag(APA, BPB, CPC, np.zeros([p, p]))
        Hx, Hu, Hv, Hz = H[0:n], H[n:n+m], H[n+m:n+m+c], H[n+m+c:n+m+c+p]
        Hxx, Hxu, Hxv, Hxz = Hx[:, 0:n], Hx[:, n:n+m], Hx[:, n+m:n+m+c], Hx[:, n+m+c:n+m+c+p]
        Hux, Huu, Huv, Huz = Hu[:, 0:n], Hu[:, n:n+m], Hu[:, n+m:n+m+c], Hu[:, n+m+c:n+m+c+p]
        Hvx, Hvu, Hvv, Hvz = Hv[:, 0:n], Hv[:, n:n+m], Hv[:, n+m:n+m+c], Hv[:, n+m+c:n+m+c+p]
        Hzx, Hzu, Hzv, Hzz = Hz[:, 0:n], Hz[:, n:n+m], Hz[:, n+m:n+m+c], Hz[:, n+m+c:n+m+c+p]

        Gx, Gu, Gv, Gz = G[0:n], G[n:n+m], G[n+m:n+m+c], G[n+m+c:n+m+c+p]
        Gxx, Gxu, Gxv, Gxz = Gx[:, 0:n], Gx[:, n:n+m], Gx[:, n+m:n+m+c], Gx[:, n+m+c:n+m+c+p]
        Gux, Guu, Guv, Guz = Gu[:, 0:n], Gu[:, n:n+m], Gu[:, n+m:n+m+c], Gu[:, n+m+c:n+m+c+p]
        Gvx, Gvu, Gvv, Gvz = Gv[:, 0:n], Gv[:, n:n+m], Gv[:, n+m:n+m+c], Gv[:, n+m+c:n+m+c+p]
        Gzx, Gzu, Gzv, Gzz = Gz[:, 0:n], Gz[:, n:n+m], Gz[:, n+m:n+m+c], Gz[:, n+m+c:n+m+c+p]

        # Control gains
        H_pre_u = -Huu + np.dot(Huv, la.solve(Hvv, Hvu))
        Ku = la.solve(H_pre_u, Hux - np.dot(Huv, la.solve(Hvv, Hvx)))
        Lu = la.solve(H_pre_u, Huz - np.dot(Huv, la.solve(Hvv, Hvz)))
        eu = la.solve(H_pre_u, np.dot((B + np.dot(C, la.solve(Hvv, Hvu))).T, q))
        # Adversary gains
        H_pre_v = -Hvv + np.dot(Hvu, la.solve(Huu, Huv))
        Kv = la.solve(H_pre_v, Hvx - np.dot(Hvu, la.solve(Huu, Hux)))
        Lv = la.solve(H_pre_v, Hvz - np.dot(Hvu, la.solve(Huu, Huz)))
        ev = la.solve(H_pre_v, np.dot((C + np.dot(B, la.solve(Huu, Huv))).T, q))

        # Record gains
        Ku_hist[t] = Ku
        Lu_hist[t] = Lu
        eu_hist[t] = eu
        Kv_hist[t] = Kv
        Lv_hist[t] = Lv
        ev_hist[t] = ev

        # Compute intermediate quantities
        IK = np.vstack([np.eye(n), Ku, Kv])
        # ABC = np.hstack([A, B, C])

        H1 = H[0:n+m+c, 0:n+m+c]
        G2 = np.block([[Gxu, Gxv, Gxz],
                       [Guu, Guv, Guz],
                       [Gvu, Gvv, Gvz]])
        G3 = np.block([[Guu, Guv, Guz],
                       [Gvu, Gvv, Gvz],
                       [Gzu, Gzv, Gzz]])

        su = np.dot(Lu, z) + eu
        sv = np.dot(Lv, z) + ev
        sz = np.hstack([su, sv, z])
        sq = np.hstack([su, sv, q])
        Bs1 = np.hstack([np.dot(B, su), np.dot(C, sv), 1])

        q_pre = np.vstack([mdot(B.T, P, A) + np.dot(mdot(B.T, P, B) + BPB, Ku),
                           mdot(C.T, P, A) + np.dot(mdot(C.T, P, C) + CPC, Kv),
                           A + np.dot(B, Ku) + np.dot(C, Kv)])

        PPqr = np.block([[P, P, q[:, None]],
                         [P, P, q[:, None]],
                         [q[None, :], q[None, :], r + np.trace(mdot(E.T, P, E, W))]])

        # Update cost function
        P_new = mdot(IK.T, H1, IK)
        q_new = mdot(IK.T, G2, sz) + np.dot(q_pre.T, sq)
        r_new = mdot(sz.T, G3, sz) + mdot(Bs1.T, PPqr, Bs1)
        P, q, r = P_new, q_new, r_new

    return Ku_hist, Lu_hist, eu_hist, Kv_hist, Lv_hist, ev_hist, P_hist, q_hist, r_hist


# this function is unused for now, it is for basic LQR problems
def compute_gains_ref_only(A_hist, B_hist, Q, R, Qf):
    n, m = Q.shape[0], R.shape[0]
    T = A_hist.shape[0]
    P_hist = np.zeros([T+1, n, n])
    K_hist = np.zeros([T, m, n])
    P = np.copy(Qf)
    for t in range(T-1, -1, -1):
        P_hist[t+1] = P
        A = A_hist[t]
        B = B_hist[t]
        Hxx = Q + mdot(A.T, P, A)
        Hux = mdot(B.T, P, A)
        Huu = R + mdot(B.T, P, B)
        K = -la.solve(Huu, Hux)
        K_hist[t] = K
        P = sympart(Hxx + np.dot(Hux.T, K))  # Take symmetric part to avoid numerical divergence issues
    return K_hist
