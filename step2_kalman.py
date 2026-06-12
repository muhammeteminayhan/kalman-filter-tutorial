import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# ----------------------------
# 1) Simülasyon (aynı)
# ----------------------------
dt = 1.0
N = 60
t = np.arange(N) * dt

v_true = 1.2
x_true = v_true * t

sigma_gps = 3.0
z = x_true + np.random.normal(0, sigma_gps, size=N)  # ölçümler

# ----------------------------
# 2) Kalman Modeli
# State: [x, v]^T
# ----------------------------
A = np.array([[1.0, dt],
              [0.0, 1.0]])          # hareket modeli
H = np.array([[1.0, 0.0]])          # sadece konumu ölçüyoruz

# Gürültüler
R = np.array([[sigma_gps**2]])      # ölçüm gürültüsü varyansı (GPS)
Q = np.array([[0.05, 0.0],          # model gürültüsü (küçük)
              [0.0,  0.05]])

# Başlangıç tahmini
x_hat = np.array([[0.0],            # konum
                  [0.0]])           # hız
P = np.array([[10.0, 0.0],
              [0.0, 10.0]])         # başlangıç belirsizliği

I = np.eye(2)

# Kayıt listeleri
x_est = np.zeros(N)
v_est = np.zeros(N)

# ----------------------------
# 3) Kalman Döngüsü
# ----------------------------
for k in range(N):
    # --- Prediction (tahmin) ---
    x_hat = A @ x_hat
    P = A @ P @ A.T + Q

    # --- Update (güncelleme) ---
    # innovation: y = z - H x_hat
    y = np.array([[z[k]]]) - (H @ x_hat)

    # innovation covariance: S = H P H^T + R
    S = H @ P @ H.T + R

    # Kalman gain: K = P H^T S^-1
    K = P @ H.T @ np.linalg.inv(S)

    # state update
    x_hat = x_hat + K @ y

    # covariance update
    P = (I - K @ H) @ P

    # kaydet
    x_est[k] = x_hat[0, 0]
    v_est[k] = x_hat[1, 0]

# ----------------------------
# 4) Görselleştirme
# ----------------------------
plt.figure()
plt.plot(t, x_true, label="Ground Truth (gerçek)")
plt.scatter(t, z, s=15, label="GPS (gürültülü ölçüm)")
plt.plot(t, x_est, label="Kalman Tahmini")
plt.xlabel("time (s)")
plt.ylabel("position x (m)")
plt.title("Step 2: Kalman filtresi GPS gürültüsünü azaltır")
plt.legend()
plt.grid(True)
plt.show()

# İstersen hız tahminini de gör:
plt.figure()
plt.plot(t, np.ones(N) * v_true, label="Gerçek hız")
plt.plot(t, v_est, label="Kalman hız tahmini")
plt.xlabel("time (s)")
plt.ylabel("velocity (m/s)")
plt.title("Kalman ile hız tahmini (sensör hız ölçmüyor!)")
plt.legend()
plt.grid(True)
plt.show()
