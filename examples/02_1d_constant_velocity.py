"""Example 02 — 1-D constant-velocity tracking.

The classic first Kalman Filter. We track an object moving at (roughly) constant
velocity, measuring **only its position**. The filter nonetheless recovers a
clean position estimate *and* infers the velocity it was never told — purely
from the motion model.

State:        x = [position, velocity]ᵀ
Measurement:  z = position
"""

import numpy as np

from _common import plt, save
from kalman import KalmanFilter

rng = np.random.default_rng(7)

dt, N = 1.0, 60
t = np.arange(N) * dt

v_true = 1.2
x_true = v_true * t
sigma_gps = 3.0
z = x_true + rng.normal(0, sigma_gps, size=N)

# --- Model -----------------------------------------------------------------
F = np.array([[1, dt],
              [0, 1]])          # constant-velocity motion
H = np.array([[1, 0]])          # we observe position only
Q = np.diag([0.05, 0.05])       # small process noise
R = np.array([[sigma_gps**2]])  # GPS variance

kf = KalmanFilter(F, H, Q, R, x0=[0, 0], P0=np.diag([10, 10]))

pos_est, vel_est = np.zeros(N), np.zeros(N)
for k in range(N):
    kf.predict()
    kf.update(z[k])
    pos_est[k], vel_est[k] = kf.state

# --- Plot ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(t, x_true, color="tab:orange", lw=2.5, label="Ground truth")
ax1.scatter(t, z, s=20, color="0.45", alpha=0.7, label="GPS measurement")
ax1.plot(t, pos_est, color="tab:blue", lw=2, label="Kalman estimate")
ax1.set(xlabel="time [s]", ylabel="position [m]", title="Position — noise filtered out")
ax1.legend()

ax2.axhline(v_true, color="tab:orange", lw=2.5, label="True velocity")
ax2.plot(t, vel_est, color="tab:blue", lw=2, label="Kalman velocity estimate")
ax2.set(xlabel="time [s]", ylabel="velocity [m/s]",
        title="Velocity — inferred, never measured")
ax2.legend()

fig.suptitle("Example 02 — 1-D constant-velocity Kalman Filter", fontsize=14)
save(fig, "02_1d_constant_velocity.png")

rmse_meas = np.sqrt(np.mean((z - x_true) ** 2))
rmse_kf = np.sqrt(np.mean((pos_est - x_true) ** 2))
print(f"  position RMSE — raw GPS: {rmse_meas:.2f} m  ->  Kalman: {rmse_kf:.2f} m")
