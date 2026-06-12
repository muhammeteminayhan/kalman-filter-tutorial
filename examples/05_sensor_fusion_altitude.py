"""Example 05 — Sensor fusion: accelerometer + barometer.

A single sensor is rarely enough:

* an **accelerometer** is fast and smooth but, integrated twice, its small bias
  makes the altitude **drift** away without bound;
* a **barometer** is absolute (no drift) but **noisy**.

The Kalman Filter fuses them: the accelerometer drives the *prediction* (as a
control input ``u``), the barometer *corrects* it. The result is both smooth and
drift-free — better than either sensor alone.

State:        x = [altitude, vertical velocity]ᵀ
Control:      u = measured vertical acceleration
Measurement:  z = barometric altitude
"""

import numpy as np

from _common import plt, save
from kalman import KalmanFilter

rng = np.random.default_rng(3)

dt, N = 0.05, 600
t = np.arange(N) * dt

# --- Ground truth: climb, cruise, descend ----------------------------------
a_true = np.zeros(N)
a_true[t < 5] = 1.5
a_true[(t >= 20) & (t < 25)] = -1.5
v_true = np.cumsum(a_true) * dt
h_true = np.cumsum(v_true) * dt

# --- Sensors ---------------------------------------------------------------
accel_bias = 0.15                                  # constant accelerometer bias
accel = a_true + accel_bias + rng.normal(0, 0.30, N)
baro = h_true + rng.normal(0, 4.0, N)              # noisy absolute altitude

# Dead-reckoning from the accelerometer alone (watch it drift):
v_dr = np.cumsum(accel) * dt
h_dr = np.cumsum(v_dr) * dt

# --- Kalman fusion ---------------------------------------------------------
F = np.array([[1, dt],
              [0, 1]])
B = np.array([[0.5 * dt**2],
              [dt]])                                # acceleration → state
H = np.array([[1, 0]])
Q = np.diag([0.02, 0.05])
R = np.array([[4.0**2]])

kf = KalmanFilter(F, H, Q, R, x0=[0, 0], P0=np.diag([5, 5]), B=B)

h_kf = np.zeros(N)
for k in range(N):
    kf.predict(u=accel[k])
    kf.update(baro[k])
    h_kf[k] = kf.state[0]

# --- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t, baro, color="0.6", lw=1, alpha=0.7, label="barometer (noisy)")
ax.plot(t, h_dr, color="tab:red", lw=1.8, ls="--", label="accelerometer only (drifts)")
ax.plot(t, h_true, color="tab:orange", lw=2.6, label="ground truth")
ax.plot(t, h_kf, color="tab:blue", lw=2, label="Kalman fusion")
ax.set(xlabel="time [s]", ylabel="altitude [m]",
       title="Example 05 — Accelerometer + barometer sensor fusion")
ax.legend()
save(fig, "05_sensor_fusion_altitude.png")

print(f"  altitude RMSE — baro: {np.sqrt(np.mean((baro-h_true)**2)):.2f} m | "
      f"accel-only: {np.sqrt(np.mean((h_dr-h_true)**2)):.2f} m | "
      f"fusion: {np.sqrt(np.mean((h_kf-h_true)**2)):.2f} m")
