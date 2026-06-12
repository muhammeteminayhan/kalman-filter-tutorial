"""A small, dependency-light **linear Kalman Filter**.

The filter estimates a hidden state ``x`` of a linear-Gaussian system from noisy
measurements ``z``. It alternates two steps:

    Predict :  x = F x + B u            (project the state forward)
               P = F P Fᵀ + Q

    Update  :  y = z − H x              (innovation / residual)
               S = H P Hᵀ + R           (innovation covariance)
               K = P Hᵀ S⁻¹             (Kalman gain)
               x = x + K y
               P = (I − K H) P          (Joseph form used internally)

Notation
--------
x : state vector                      (n, 1)
P : state covariance                  (n, n)
F : state-transition matrix           (n, n)
B : control matrix (optional)         (n, m)
H : measurement matrix                (k, n)
Q : process-noise covariance          (n, n)
R : measurement-noise covariance      (k, k)
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """A general n-dimensional linear Kalman Filter."""

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F = np.atleast_2d(F).astype(float)
        self.H = np.atleast_2d(H).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.B = None if B is None else np.atleast_2d(B).astype(float)

        self.x = np.reshape(np.asarray(x0, dtype=float), (-1, 1))
        self.P = np.atleast_2d(P0).astype(float)

        self.n = self.F.shape[0]
        self._I = np.eye(self.n)

    # ------------------------------------------------------------------ #
    def predict(self, u=None):
        """Project the state and covariance one step ahead.

        ``u`` is an optional control input (e.g. a measured acceleration).
        """
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.reshape(np.asarray(u, float), (-1, 1))
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    # ------------------------------------------------------------------ #
    def update(self, z):
        """Correct the prediction with a measurement ``z``."""
        z = np.reshape(np.asarray(z, dtype=float), (-1, 1))

        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y

        # Joseph-form covariance update — numerically stable and stays symmetric.
        A = self._I - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.x

    # ------------------------------------------------------------------ #
    @property
    def state(self):
        """Current state estimate as a flat 1-D array."""
        return self.x.ravel()
