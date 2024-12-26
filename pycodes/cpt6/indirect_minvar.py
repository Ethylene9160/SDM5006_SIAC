# This is a python implementation of example622.m.
# Applying  self tuning control (minimum variance) while doing identification.
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import diophantine as dp

a = np.array([1, -1.7, 0.7])
b = np.array([1, 0.5])
c = np.array([1, 0.2])
d = 4

steps = 400


# white noise with 0.1 variance
v = np.random.normal(1, 0.1, steps)

# input signal
r = 10
yr = r * np.array([np.ones(int(steps/4)), 
                    -np.ones(int(steps/4)),
                    np.ones(int(steps/4)), 
                    -np.ones(int(steps/4))]).flatten()

theta = 0.001*np.ones((5, 1))

u_records = np.zeros(steps)
y_records = np.zeros(steps)

# y0-y4:
y_records[0] = v[0]
y_records[1] = 1.7 * y_records[0] + v[1] + 0.2*v[0]
y_records[2] = 1.7 * y_records[1] - 0.7 * y_records[0] + v[2] + 0.2*v[1]
y_records[3] = 1.7 * y_records[2] - 0.7 * y_records[1] + v[3] + 0.2*v[2]
y_records[4] = 1.7 * y_records[3] - 0.7 * y_records[2] + v[4] + 0.2*v[3]

u_k4 = u_k5 = 0
P = 100000000 * np.eye(5)
for k in range(d+1, 10):
    phi = np.array([-y_records[k], -y_records[k-1], u_records[k-3], u_records[k-4], v[k]]).reshape(5,1)
    K = P @ phi / (1 + phi.T @ P @ phi)
    theta = theta + K * (yr[k] - phi.T @ theta)
    P = (P - K @ phi.T @ P) # / _lambda
    print('theta: ', theta)

    # estimate a, b, c:
    ae = np.array([1, -theta[0, 0], -theta[1, 0]])
    be = np.array([theta[2, 0], theta[3, 0]])
    ce = np.array([1, theta[4, 0]])
    if abs(be[1]) > 0.9:
        be[1] = np.sign(ce[1]) * 0.9
    if abs(ce[1]) > 0.9:
        ce[1] = np.sign(ce[1]) * 0.9

    e, f, g = dp.sindiophantine(a, b, c, d)
    u = (-f[1:] @ np.array([u_records[k], u_records[k-1], u_records[k-2], u_records[k-3]]).reshape(4,1) \
        + c @ np.array([yr[k], yr[k-1]]).reshape(2,1) - g @ np.array([y_records[k], y_records[k-1]]).reshape(2,1))/f[0]
    print('u: ', u)
    u_records[k] = u.item()
    # plant output
    y_k = -a[1:] @ y_records[k-1:k-3:-1] + b @ np.array([u_records[k-4], u_records[k-5]]) + c @ np.array([v[k], v[k-1]]).reshape(2,1)
    y_k = y_k.item()
    print(y_k)
    y_records[k] = y_k

    # recursive generalised least square method
    # # control input
    # u_k = -theta[1:] @ np.array([u_records[k-1], u_records[k-2], u_records[k-3], u_records[k-4], u_k4, u_k5]).reshape(5,1) + c @ np.array([yr[k], yr[k-1]]).reshape(2,1)
    # u_k = u_k.item()
    # u_records[k] = u_k
    # u_k5 = u_k4
    # u_k4 = u_k
plt.figure()
plt.subplot(211)
plt.plot(np.arange(0, steps), yr, label='input signal (yr)', color='black')
plt.plot(np.arange(0, steps), y_records, label='plant output (y)')
plt.legend()
plt.title('Minimum Variance Identification')
plt.subplot(212)
plt.plot(np.arange(0, steps), u_records, label='control input (u)')
plt.legend()
plt.title('Control Input')
plt.show()