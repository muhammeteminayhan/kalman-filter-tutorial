"""An **Extended Kalman Filter (EKF)** for mildly non-linear systems.

When the motion model ``f(x, u)`` or the measurement model ``h(x)`` is
non-linear, the plain Kalman Filter no longer applies directly. The EKF
linearises both models at the current estimate using their Jacobians
(``F = ∂f/∂x`` and ``H = ∂h/∂x``) and then runs the usual predict/update
equations on that local linearisation.

    Predict :  x = f(x, u)
               P = F P Fᵀ + Q            with F = ∂f/∂x |ₓ

    Update  :  y = z − h(x)
               S = H P Hᵀ + R            with H = ∂h/∂x |ₓ
               K = P Hᵀ S⁻¹
               x = x + K y
               P = (I − K H) P
"""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    """EKF parameterised by callables for the models and their Jacobians.

    Parameters
    ----------
    f       : callable ``f(x, u) -> (n, 1)``      non-linear state transition
    F_jac   : callable ``F_jac(x, u) -> (n, n)``  Jacobian of ``f`` w.r.t. ``x``
    h       : callable ``h(x) -> (k, 1)``         non-linear measurement model
    H_jac   : callable ``H_jac(x) -> (k, n)``     Jacobian of ``h`` w.r.t. ``x``
    Q, R    : process / measurement noise covariances
    x0, P0  : initial state and covariance
    """

    def __init__(self, f, F_jac, h, H_jac, Q, R, x0, P0):
        self.f = f
        self.F_jac = F_jac
        self.h = h
        self.H_jac = H_jac
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.x = np.reshape(np.asarray(x0, dtype=float), (-1, 1))
        self.P = np.atleast_2d(P0).astype(float)
        self.n = self.x.shape[0]
        self._I = np.eye(self.n)

    # ------------------------------------------------------------------ #
    def predict(self, u=None):
        F = np.atleast_2d(self.F_jac(self.x, u)).astype(float)
        self.x = np.reshape(self.f(self.x, u), (-1, 1))
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    # ------------------------------------------------------------------ #
    def update(self, z):
        z = np.reshape(np.asarray(z, dtype=float), (-1, 1))
        H = np.atleast_2d(self.H_jac(self.x)).astype(float)

        y = z - np.reshape(self.h(self.x), (-1, 1))
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        A = self._I - K @ H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.x

    @property
    def state(self):
        return self.x.ravel()
