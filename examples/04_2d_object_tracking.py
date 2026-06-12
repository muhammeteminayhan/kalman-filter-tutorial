"""Example 04 — 2-D object tracking.

A target moves along a curved path on a plane. A position sensor (e.g. radar
plots converted to x/y) returns noisy ``(x, y)`` readings. A 4-state constant-
velocity Kalman Filter smooths the track and estimates the velocity vector.

State:        x = [x, vx, y, vy]ᵀ
Measurement:  z = [x, y]
"""

import numpy as np

from _common import plt, save
from kalman import KalmanFilter

rng = np.random.default_rng(42)

dt, N = 0.1, 250
t = np.arange(N) * dt

# Curved ground truth: forward motion + a sweeping turn.
x_true = 6.0 * t
y_true = 18.0 * np.sin(0.25 * t)
truth = np.column_stack([x_true, y_true])

sigma = 1.6
meas = truth + rng.normal(0, sigma, size=truth.shape)

# --- Model (constant velocity, decoupled x and y) --------------------------
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Q = np.diag([0.05, 0.5, 0.05, 0.5])
R = np.diag([sigma**2, sigma**2])

kf = KalmanFilter(F, H, Q, R, x0=[0, 0, 0, 0], P0=np.diag([10, 10, 10, 10]))

est = np.zeros((N, 2))
for k in range(N):
    kf.predict()
    kf.update(meas[k])
    s = kf.state
    est[k] = [s[0], s[2]]

# --- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(meas[:, 0], meas[:, 1], s=12, color="0.55", alpha=0.5, label="measurements")
ax.plot(x_true, y_true, color="tab:orange", lw=2.5, label="ground truth")
ax.plot(est[:, 0], est[:, 1], color="tab:blue", lw=2, label="Kalman estimate")
ax.set(xlabel="x [m]", ylabel="y [m]", title="Example 04 — 2-D constant-velocity tracking")
ax.axis("equal")
ax.legend()
save(fig, "04_2d_object_tracking.png")

rmse_meas = np.sqrt(np.mean(np.sum((meas - truth) ** 2, axis=1)))
rmse_kf = np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1)))
print(f"  track RMSE — raw: {rmse_meas:.2f} m  ->  Kalman: {rmse_kf:.2f} m")
