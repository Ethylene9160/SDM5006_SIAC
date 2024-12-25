import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# impulse response for system identification
# y(k) = 0.16u(k) + 0.12u(k-1) + 0.10u(k-2) + u(k)
g = np.array([0.16, 0.12, 0.10])
N = 10
t = np.linspace(0, N, N)
u = np.zeros(N)
u[0] = 1

# Compute the response of the system
y = np.zeros(N)
for i in range(N):
    for j in range(i + 1):
        y[i] += g[j] * u[i - j]

# display the response
plt.figure()
plt.stem(t, y)
plt.title('Impulse Response of the System')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.grid()
plt.show()