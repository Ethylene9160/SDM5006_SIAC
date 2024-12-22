# This is an implementation of example621.m.
import diophantine as dp
import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, -1.7, 0.7])
b = np.array([1, 0.5])
c = np.array([1, 0.2])
d = 4

na = len(a)-1
nb = len(b)-1
nc = len(c)-1
nf = nb + d - 1

e, f, g = dp.sindiophantine(a, b, c, d)
print('e: ', e)
print('f: ', f)
print('g: ', g)
timestamps = 400

u = np.zeros((d+nb, 1))
y = np.zeros((na, 1))
yr = np.zeros((nc, 1))
xi = np.zeros((nc, 1))

y_records = [0,0,0,0]
u_records = [0,0,0,0]

setpoints = 10 * np.array([np.ones(int(timestamps/4)), 
                    -np.ones(int(timestamps/4)),
                    np.ones(int(timestamps/4)), 
                    -np.ones(int(timestamps/4))]).flatten()
v = np.random.normal(0, 0.1, timestamps)

for i in range(3, timestamps-4):
    # plant output
    # y_k = -a[1:] @ y + b @ u[d-1:] + c @ np.array([v[i], v[i-1]]).reshape(2,1)
    y_k = -a[1:] @ y + b @ u[d-1:] + c @ np.array([v[i], v[i-1]]).reshape(2,1)
    # print('shape of yk: ', y_k.shape)
    # print('yk: ', y_k)
    y_k = y_k.item()
    y_records.append(y_k)
    # u_k = -f[1:] @ u[:nf]+ c @ setpoints[i+d:i+d-1-min(d, nc):-1].reshape(2,1)
    u_k = -f[1:] @ np.array(
        [u_records[i],
         u_records[i-1],
         u_records[i-2],
         u_records[i-3]]).reshape((4,1)) +\
            c @ np.array([setpoints[i+4],setpoints[i+3]]).reshape(2,1) -\
            g @ np.array([y_records[i], y_records[i-1]]).reshape(2,1)
    # print('shape of uk: ', u_k.shape)
    # print('uk: ', u_k)
    u_k = u_k.item()
    u_records.append(u_k)
    # u1 = u0
    # u2 = u1
    # u[-1] = u_k
    for j in range(1, d):
        u[j-1] = u[j]
    u[d-1] = u_k

    # y1 = y0
    # y2 = y1
    # y[-1] = y_k
    for j in range(1, na):
        y[j-1] = y[j]
    y[na-1] = y_k

plt.figure()
plt.plot(np.arange(0, len(y_records)-1, 1), setpoints[:-d], label='setpoints')
plt.plot(np.arange(0, len(y_records)-1, 1), y_records[1:], label='plant output')
plt.legend()
plt.show()

