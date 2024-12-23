# This is an implementation of stable adaptive control 
# whose transfer function is G(s)= (s+1)/(s^2-5s+6).
# This is a python version for example541.m
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# plant transfer function
nump = np.array([1, 1])
denp = np.array([1, -5, 6])
Ap, Bp, Cp, Dp = signal.tf2ss(nump, denp)
# print("Ap: ", Ap)
# print("Bp: ", Bp)
# print("Cp: ", Cp)
# print("Dp: ", Dp)

# model transfer function
numm = np.array([1, 2])
denm = np.array([1, 3, 6])
Am, Bm, Cm, Dm = signal.tf2ss(numm, denm)
Df=numm; # Denominator polynomial of the transfer function 

# order of the transfer function
n = len(denp) - 1

dt = 0.01
simulation_time = 40
time_length = int(simulation_time/dt)

Af = np.zeros((n-1, n-1))
Af[1:, :-1] = np.eye(n-2)
Af[-1, :]=-np.array([Df[1:]])
print("Af: ", Af)
Bf = np.zeros((n-1, 1))
Bf[-1, 0] = 1
print("Bf: ", Bf)

yr0 = 0
yp0 = 0
u0 = 0
e0 = 0
v10 = np.zeros((n-1, 1))
v20 = np.zeros((n-1, 1))
xp0 = np.zeros((n, 1))
xm0 = np.zeros((n, 1))
theta0 = np.zeros(( 2 * n,1))

r = 2
yr = r * np.array([np.ones(int(time_length/4)), 
                    -np.ones(int(time_length/4)),
                    np.ones(int(time_length/4)), 
                    -np.ones(int(time_length/4))]).flatten()
Gamma = 10*np.eye(2 * n)


xp = xp0.copy()
xm = xm0.copy()
u = u0

v1 = v10.copy()
v2 = v20.copy()

theta = theta0.copy()

yp_records = np.zeros(time_length)
ym_records = np.zeros(time_length)
u_records = np.zeros(time_length)
for k in range(time_length):
    # plant output
    yp = Cp @ xp + Dp * u
    xp_dot = Ap @ xp + Bp * u
    xp = xp + dt * xp_dot
    yp_records[k] = yp.item()

    # model output
    ym = Cm @ xm + Dm * yr[k]
    xm_hat = Am @ xm + Bm * yr[k]
    xm = xm + dt * xm_hat
    ym_records[k] = ym.item()

    # error
    e = ym - yp

    # v1, v2
    v1_dot = Af @ v1 + Bf * u
    v1 = v1 + dt * v1_dot
    v2_dot = Af @ v2 + Bf * yp
    v2 = v2 + dt * v2_dot
    phi = np.concatenate((yr[k].flatten(), v1.flatten(), yp.flatten(), v2.flatten()))
    
    phi = phi.reshape((2 * n,1))
    # print('phi: ', phi)
    # print('Gamma: ', Gamma)
    theta = theta + dt * e * Gamma @ phi
    # print('theta: ', theta)
    u = (theta.T @ phi).item()
    u_records[k] = u


plt.figure()
plt.subplot(211)
# plt.plot(np.arange(0, simulation_time, dt), yr, label='input signal (yr)', color='black')
plt.plot(np.arange(0, simulation_time, dt), ym_records, label='model output (ym)')
plt.plot(np.arange(0, simulation_time, dt), yp_records, label='plant output (yp)')
plt.xlabel('time')
plt.ylabel('output')
plt.legend()
plt.title('Stable Adaptive Control')
plt.subplot(212)
plt.plot(np.arange(0, simulation_time, dt), u_records, label='control input (u)')
plt.xlabel('time')
plt.ylabel('control input')
plt.legend()
plt.show()
