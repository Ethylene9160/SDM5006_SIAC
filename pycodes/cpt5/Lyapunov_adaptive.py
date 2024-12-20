# This is an implementation of the Lyapunov-based adaptive control for the
# Based on example522.m

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# simulation config
T = 40
dt = 0.01
timestampes = np.arange(0, T, dt)
timeLength = len(timestampes)

# Define the system matrices
Ap = np.array([[0, 1], [-6, -7]])
Bp = np.array([[0], [8]])

Am = np.array([[0, 1], [-10, -5]])
Bm = np.array([[0], [2]])

n = 2
m = 1

P = np.array([[3, 1], [1, 1]]) # positive definite matrix
R1 = 1.0 * np.eye(m)
R2 = 1.0 * np.eye(m)

u_records = np.zeros(timeLength)
yr_recods = np.zeros(timeLength)
e_records = []
xp_records = []
xm_records = []
xp = np.zeros((n,1))
xm = np.zeros((n,1))

F = np.zeros((m, n))
K = np.zeros(m)
e = np.zeros((n, 1))
for i in range(timeLength):

    t = timestampes[i]
    yr = 1.0 * (0.01*np.pi*t+4*np.sin(0.2*np.pi*t)+np.sin(np.pi*t))
    yr_recods[i] = yr

    F = F + dt * (R1.dot(Bm.T).dot(P).dot(e)*(xp.T))
    K = K + dt * (R2.dot(Bm.T).dot(P).dot(e)*(yr))
    # print('shpa of F: ', F.shape)
    # print('shape of K: ', K.shape)
    u = F.dot(xp) + K.dot(yr)
    # print('shape of u: ', u.shape)
    u_records[i] = u
    # Compute the control input
    xp = xp + dt * (Ap.dot(xp) + Bp.dot(u))
    xm = xm + dt * (Am.dot(xm) + Bm.dot(yr))
    xp_records.append(xp)
    xm_records.append(xm)

    e = xm - xp
    e_records.append(e)
xp_records = np.array(xp_records)
xm_records = np.array(xm_records)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(timestampes, xp_records[:, 0], label=r'$x_{p1}(t)$')
plt.plot(timestampes, xm_records[:, 0], label=r'$x_{m1}(t)$')
plt.legend()
plt.title(r'$x_{p1}(t),~x_{m1}(t)$')
plt.subplot(2, 1, 2)
plt.plot(timestampes, xp_records[:, 1], label=r'$x_{p2}(t)$')
plt.plot(timestampes, xm_records[:, 1], label=r'$x_{m2}(t)$')
plt.legend()
plt.title(r'$x_{p2}(t),~x_{m2}(t)$')
plt.tight_layout()
plt.show()

