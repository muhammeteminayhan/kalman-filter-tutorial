import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# 1) zaman
dt = 1.0
N = 60
t = np.arange(N) * dt

# 2) gerçeği üret (sabit hız)
v_true = 1.2  # m/s
x_true = v_true * t  # x = v*t

# 3) GPS ölçümü üret (gürültü ekle)
sigma_gps = 3.0  # metre (sensör hatası)
gps = x_true + np.random.normal(0, sigma_gps, size=N)

# 4) görselleştir
plt.figure()
plt.plot(t, x_true, label="Ground Truth (gerçek)")
plt.scatter(t, gps, s=15, label="GPS (gürültülü ölçüm)")
plt.xlabel("time (s)")
plt.ylabel("position x (m)")
plt.title("Step 1: Gürültülü ölçüm nedir?")
plt.legend()
plt.grid(True)
plt.show()
