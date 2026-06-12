# `data/`

Example **07** runs on a real tethered-drone flight log. The raw CSVs are **not**
committed (they are large and separately licensed), so this folder ships empty.

To reproduce `figures/07_realdata_flight_altitude.png`, drop these files here:

```
data/
├── BarometricPressure.csv   # Timpstemp, Pressure, Altitude, Temperature, ...
├── OnboardPose.csv          # IMU (Accel_*, Attitude quaternion), 50 Hz
├── GroundTruthAGL.csv       # imgid, x_gt, y_gt, z_gt, x_gps, y_gps, z_gps, ...
└── GroundTruthAGM.csv       # Timpstemp, imgid, ...  (maps image-ids to time)
```

If the files are absent, `examples/07_realdata_flight_altitude.py` simply prints a
notice and exits — every other example is fully self-contained and needs no data.
