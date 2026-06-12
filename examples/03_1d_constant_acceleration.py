"""Example 03 — Constant-acceleration model.

When the target speeds up, a constant-velocity model lags behind. Adding
acceleration to the state lets the filter follow manoeuvres and even estimate
the (unmeasured) acceleration from position measurements alone.

State:        x = [position, velocity, acceleration]ᵀ
Measurement:  z = position
"""

import numpy as np

from _common import plt, save
from kalman import KalmanFilter

rng = np.random.default_rng(1)

dt, N = 0.1, 200
t = np.arange(N) * dt

a_true = 0.8                                  # constant acceleration [m/s²]
x_true = 0.5 * a_true * t**2                  # x = ½ a t²
v_true = a_true * t

sigma_pos = 2.0
z = x_true + rng.normal(0, sigma_pos, size=N)

# --- Model -----------------------------------------------------------------
F = np.array([[1, dt, 0.5 * dt**2],
              [0,  1, dt],
              [0,  0, 1]])                    # constant-acceleration kinematics
H = np.array([[1, 0, 0]])                     # measure position only
Q = np.diag([1e-3, 1e-3, 1e-2])
R = np.array([[sigma_pos**2]])

kf = KalmanFilter(F, H, Q, R, x0=[0, 0, 0], P0=np.diag([10, 10, 10]))

est = np.zeros((N, 3))
for k in range(N):
    kf.predict()
    kf.update(z[k])
    est[k] = kf.state

# --- Plot ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(t, x_true, color="tab:orange", lw=2.5, label="truth")
axes[0].scatter(t, z, s=8, color="0.5", alpha=0.5, label="measurement")
axes[0].plot(t, est[:, 0], color="tab:blue", lw=2, label="estimate")
axes[0].set(xlabel="time [s]", ylabel="position [m]", title="Position")
axes[0].legend()

axes[1].plot(t, v_true, color="tab:orange", lw=2.5, label="truth")
axes[1].plot(t, est[:, 1], color="tab:blue", lw=2, label="estimate")
axes[1].set(xlabel="time [s]", ylabel="velocity [m/s]", title="Velocity (inferred)")
axes[1].legend()

axes[2].axhline(a_true, color="tab:orange", lw=2.5, label="truth")
axes[2].plot(t, est[:, 2], color="tab:blue", lw=2, label="estimate")
axes[2].set(xlabel="time [s]", ylabel="acceleration [m/s²]", title="Acceleration (inferred)")
axes[2].legend()

fig.suptitle("Example 03 — Constant-acceleration Kalman Filter", fontsize=14)
save(fig, "03_1d_constant_acceleration.png")
