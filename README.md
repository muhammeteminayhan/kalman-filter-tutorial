# Kalman Filter Basics

A minimal, self-contained **1-D Kalman Filter** example in Python — a clean reference for state estimation under noisy measurements (constant-velocity model).

## 📜 Scripts

| Script | What it does |
| --- | --- |
| [`step1_simulation.py`](step1_simulation.py) | Simulates ground-truth motion (constant velocity) and generates **noisy position measurements** |
| [`step2_kalman.py`](step2_kalman.py) | Applies a **Kalman Filter** to the noisy measurements to estimate position & velocity, then plots truth vs. measurement vs. estimate |

## 🧠 Concept

The filter tracks a 2-state vector **[position, velocity]** through the predict/update cycle:

- **Predict:** project the state forward with the motion model (`x = F·x`, `P = F·P·Fᵀ + Q`)
- **Update:** correct the prediction with each measurement via the Kalman gain (`K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹`)

## ▶️ Running

```bash
pip install numpy matplotlib
python step1_simulation.py
python step2_kalman.py
```

## 📄 License

Released under the [MIT License](LICENSE).
