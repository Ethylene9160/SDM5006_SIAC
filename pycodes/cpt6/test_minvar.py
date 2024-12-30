import numpy as np
import matplotlib.pyplot as plt
import diophantine as dp

timestamps = 400

a = np.array([1, -1.7, 0.7])
b = np.array([1, 0.5])
c = np.array([1, 0.2])
d = 4

a_hat = np.array([1, -1.7, 0.7])
b_hat = np.array([1, 0.5])
c_hat = np.array([1, 0.2])

setpoints = 10 * np.array([
                    np.ones(int(timestamps/4)), 
                    -np.ones(int(timestamps/4)),
                    np.ones(int(timestamps/4)), 
                    -np.ones(int(timestamps/4))]).flatten()
setpoints = np.concatenate((np.zeros(d+2), setpoints, -np.ones(d+1)))
v = 0.1 * np.random.normal(0, 0.1, timestamps+d+1)

y = np.zeros((d+2+timestamps, 1))
u = np.ones((d+2+timestamps, 1))
u[0]=0
e, f, g = dp.sindiophantine(a, b, c, d)
print('e: ', e)
print('f: ', f)
print('g: ', g)

for i in range(d+2, timestamps+d+2):
    print('i: ', i)
    print('u[i-4]: ', u[i-4])
    print('u[i-5]: ', u[i-5])
    print('u:', u[i-4:i-6:-1])
    y[i] = -a[1:] @ y[i-1:i-3:-1] + b @ u[i-4:i-6:-1] + c @ v[i-1:i-3:-1]
    u[i] = -f[1:] @ u[i-1:i-5:-1] + c @ setpoints[i+d:i+d-2:-1] - g @ y[i:i-2:-1]
    u[i] /= f[0]

plt.figure()
plt.plot(y[d+2:], label='y')
plt.plot(setpoints[d+2:-d-1], label='setpoints')
plt.legend()
plt.show()

