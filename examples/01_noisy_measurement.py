"""Example 01 — What is measurement noise?

Before filtering anything, it helps to *see* the problem. A sensor (here a GPS)
reports the position of an object moving at constant velocity, but every reading
is corrupted by random noise. The goal of the whole tutorial is to recover the
smooth truth (orange) from the noisy dots (grey).
"""

import numpy as np

from _common import plt, save

rng = np.random.default_rng(7)

dt, N = 1.0, 60
t = np.arange(N) * dt

v_true = 1.2                     # m/s
x_true = v_true * t              # straight line: x = v·t

sigma_gps = 3.0                  # GPS standard deviation [m]
z = x_true + rng.normal(0, sigma_gps, size=N)

fig, ax = plt.subplots()
ax.plot(t, x_true, color="tab:orange", lw=2.5, label="Ground truth")
ax.scatter(t, z, s=22, color="0.45", alpha=0.8, label="GPS measurement (noisy)")
ax.set_xlabel("time [s]")
ax.set_ylabel("position x [m]")
ax.set_title("Example 01 — A noisy sensor measurement")
ax.legend()
save(fig, "01_noisy_measurement.png")
