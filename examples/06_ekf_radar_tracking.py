"""Example 06 — Extended Kalman Filter: radar tracking.

A target flies past a **radar** sitting at the origin. The radar does not measure
``(x, y)`` directly — it measures **range** and **bearing**, which are *non-linear*
functions of the state:

    range(x, y)   = √(x² + y²)
    bearing(x, y) = atan2(y, x)

The motion is linear (constant velocity) but the measurement is not, so we use an
**Extended Kalman Filter**, linearising the measurement model through its Jacobian
at every step.

State:        x = [x, vx, y, vy]ᵀ
Measurement:  z = [range, bearing]
"""

import numpy as np

from _common import plt, save
from kalman import ExtendedKalmanFilter

rng = np.random.default_rng(11)

dt, N = 1.0, 25
t = np.arange(N) * dt

# --- Ground truth (constant velocity, passes in front of the radar) --------
x0, vx, y0, vy = -100.0, 8.0, 40.0, 2.0
px = x0 + vx * t
py = y0 + vy * t

# --- Non-linear measurement model and its Jacobian -------------------------
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])


def f(x, u=None):
    return F @ x


def F_jac(x, u=None):
    return F


def h(x):
    px, py = x[0, 0], x[2, 0]
    return np.array([[np.hypot(px, py)],
                     [np.arctan2(py, px)]])


def H_jac(x):
    px, py = x[0, 0], x[2, 0]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    return np.array([[px / r, 0, py / r, 0],
                     [-py / r2, 0, px / r2, 0]])


sigma_r, sigma_b = 5.0, 0.03                       # range [m], bearing [rad]
R = np.diag([sigma_r**2, sigma_b**2])
Q = np.diag([0.1, 0.1, 0.1, 0.1])

# Noisy radar measurements
rng_meas = np.hypot(px, py) + rng.normal(0, sigma_r, N)
brg_meas = np.arctan2(py, px) + rng.normal(0, sigma_b, N)

ekf = ExtendedKalmanFilter(
    f, F_jac, h, H_jac, Q, R,
    x0=[-90, 0, 30, 0], P0=np.diag([100, 50, 100, 50]),
)

est = np.zeros((N, 2))
for k in range(N):
    ekf.predict()
    ekf.update([rng_meas[k], brg_meas[k]])
    s = ekf.state
    est[k] = [s[0], s[2]]

# Convert raw measurements to x/y just for visualisation
mx = rng_meas * np.cos(brg_meas)
my = rng_meas * np.sin(brg_meas)

# --- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(0, 0, marker="^", s=160, color="k", label="radar", zorder=5)
ax.scatter(mx, my, s=22, color="0.55", alpha=0.6, label="radar measurements (→ x/y)")
ax.plot(px, py, color="tab:orange", lw=2.6, label="ground truth")
ax.plot(est[:, 0], est[:, 1], color="tab:blue", lw=2, marker="o", ms=3, label="EKF estimate")
ax.set(xlabel="x [m]", ylabel="y [m]",
       title="Example 06 — EKF tracking from range + bearing")
ax.axis("equal")
ax.legend()
save(fig, "06_ekf_radar_tracking.png")

truth = np.column_stack([px, py])
print(f"  position RMSE — EKF: {np.sqrt(np.mean(np.sum((est-truth)**2, axis=1))):.2f} m")
