"""Example 07 — Real flight log: altitude estimation (case study).

This applies the filter to a **real tethered-drone flight log** (~45 min, Zurich).
The aircraft carries several altitude-related sensors:

* a **barometer**     — precise, ~9 Hz, but with high-frequency jitter;
* an **IMU**          — 50 Hz accelerometer + attitude (used to predict);
* a **GPS** receiver  — absolute, but several metres off (bias + noise).

We fuse the barometer (measurement) with the IMU vertical acceleration (control
input) in a 2-state Kalman Filter and compare the result against the survey-grade
**ground truth**. The fused estimate tracks ground truth within a few metres and
is far closer than raw GPS.

    State:        x = [altitude, vertical velocity]ᵀ
    Control:      u = world-frame vertical acceleration (from IMU + attitude)
    Measurement:  z = barometric altitude

The raw flight log is **not** committed to this repository (size + licensing).
Place the CSVs under ``data/`` to reproduce — otherwise this script exits cleanly.
"""

import os

import numpy as np

from _common import ROOT, plt, save

DATA = os.path.join(ROOT, "data")
NEEDED = ["BarometricPressure.csv", "OnboardPose.csv",
          "GroundTruthAGL.csv", "GroundTruthAGM.csv"]


def _missing():
    return [f for f in NEEDED if not os.path.exists(os.path.join(DATA, f))]


def _load(name):
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA, name))
    d.columns = [c.strip() for c in d.columns]
    return d


def main():
    miss = _missing()
    if miss:
        print("  [skip] flight log not found in data/ — missing:", ", ".join(miss))
        print("         Drop the CSVs into data/ to reproduce this figure.")
        return

    import pandas as pd  # noqa: F401  (only needed when data is present)

    baro = _load("BarometricPressure.csv").sort_values("Timpstemp")
    pose = _load("OnboardPose.csv").sort_values("Timpstemp")
    agm = _load("GroundTruthAGM.csv")
    gt = _load("GroundTruthAGL.csv")

    # Ground truth has no timestamp; recover it from the AGM (image-id → time) table.
    ts_of_img = agm.groupby("imgid")["Timpstemp"].min()
    gt = gt.assign(ts=gt["imgid"].map(ts_of_img)).dropna(subset=["ts"]).sort_values("ts")

    t0 = min(baro.Timpstemp.min(), pose.Timpstemp.min())
    to_s = lambda v: (np.asarray(v) - t0) / 1e6     # microseconds → seconds

    bt, ba = to_s(baro.Timpstemp.values), baro.Altitude.values
    pt = to_s(pose.Timpstemp.values)
    gtt, gz, gps = to_s(gt.ts.values), gt.z_gt.values, gt.z_gps.values

    # --- World-frame vertical acceleration from body IMU + attitude quaternion ---
    qw, qx, qy, qz = (pose[c].values for c in
                      ("Attitude_w", "Attitude_x", "Attitude_y", "Attitude_z"))
    axb, ayb, azb = (pose[c].values for c in ("Accel_x", "Accel_y", "Accel_z"))
    # third row of the body→world rotation matrix R(q), dotted with body accel:
    a_world_z = (2 * (qx * qz - qw * qy) * axb
                 + 2 * (qy * qz + qw * qx) * ayb
                 + (1 - 2 * (qx**2 + qy**2)) * azb)
    a_vert = a_world_z - np.median(a_world_z)        # remove gravity → vertical accel
    a_on_baro = np.interp(bt, pt, a_vert)            # resample to the baro clock

    # --- Kalman Filter: barometer (measurement) + IMU accel (control) ---
    from kalman import KalmanFilter
    R = np.array([[6.0**2]])
    Q = np.diag([0.02, 0.05])
    kf = KalmanFilter(np.eye(2), np.array([[1, 0]]), Q, R,
                      x0=[ba[0], 0], P0=np.diag([10, 10]))

    est = np.zeros(len(bt))
    for k in range(len(bt)):
        dt = bt[k] - bt[k - 1] if k else 0.1
        if not (0 < dt < 1):
            dt = 0.1
        kf.F = np.array([[1, dt], [0, 1]])
        kf.B = np.array([[0.5 * dt**2], [dt]])
        kf.predict(u=a_on_baro[k])
        kf.update(ba[k])
        est[k] = kf.state[0]

    # --- Accuracy vs ground truth ---
    m = (bt >= gtt.min()) & (bt <= gtt.max())
    gz_on_bt = np.interp(bt, gtt, gz)
    rmse = lambda a: np.sqrt(np.mean((a[m] - gz_on_bt[m]) ** 2))
    gps_rmse = np.sqrt(np.mean((gps - gz) ** 2))
    print(f"  RMSE vs ground truth — raw GPS: {gps_rmse:.2f} m | "
          f"Kalman (baro+IMU): {rmse(est):.2f} m")

    # --- Plot: full flight + dynamic zoom ---
    z0, z1 = 1680, 1800                              # most dynamic 2-min window
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(15, 5.5))

    axf.scatter(gtt, gps, s=10, color="0.6", alpha=0.5, label="raw GPS (noisy)")
    axf.plot(bt, est, color="tab:blue", lw=1.4, label="Kalman (baro + IMU)")
    axf.plot(gtt, gz, color="tab:orange", lw=2, label="ground truth")
    axf.set(xlabel="time [s]", ylabel="altitude [m]",
            title=f"Full flight (~45 min) — KF {rmse(est):.1f} m vs GPS {gps_rmse:.1f} m")
    axf.axvspan(z0, z1, color="tab:blue", alpha=0.07)
    axf.legend(loc="upper right")

    zb = (bt >= z0) & (bt <= z1)
    zg = (gtt >= z0) & (gtt <= z1)
    axz.plot(bt[zb], ba[zb], color="0.65", lw=1, label="raw barometer (jittery)")
    axz.plot(bt[zb], est[zb], color="tab:blue", lw=2, label="Kalman estimate (smooth)")
    axz.scatter(gtt[zg], gz[zg], s=28, color="tab:orange", zorder=5, label="ground truth")
    axz.set(xlabel="time [s]", ylabel="altitude [m]",
            title="Zoom: 2-minute dynamic segment")
    axz.legend(loc="best")

    fig.suptitle("Example 07 — Real tethered-drone flight log: altitude estimation",
                 fontsize=14)
    save(fig, "07_realdata_flight_altitude.png")


if __name__ == "__main__":
    main()
