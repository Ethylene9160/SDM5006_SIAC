# This is an override version for example511.m,
# a python version for model based machine learning.
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def cal_u(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def get_u(self, errors):
        return self.kp*errors[0] + self.ki*errors[1] + self.kd*errors[2]


class error_store:
    def __init__(self):
        self.ep = 0
        self.ei = 0
        self.ed = 0

    def update(self, error, dt):
        self.ed = (error - self.ep) / dt
        self.ep = error
        self.ei += error*dt
        return [self.ep, self.ei, self.ed]

# Transfer function
num = np.array([1])
den = np.array([1, 1, 1])
n = len(den) - 1

# simulate time
dt = 0.1
simulate_time = 100

kp = 1.0
km = 1.0

# calculate A, B, C, D matrix
plant_k = signal.TransferFunction(kp * num, den)
plant_m = signal.TransferFunction(km * num, den)

Ap, Bp, Cp, Dp = signal.tf2ss(plant_k.num, plant_k.den)
Am, Bm, Cm, Dm = signal.tf2ss(plant_m.num, plant_m.den)

print('Ap: ', Ap)
print('Bp: ', Bp)
print('Cp: ', Cp)
print('Dp: ', Dp)

gamma = 0.1 # adaptive gain
u = 0 # control input, by the controller.
xp = np.zeros((n,1)) # plant state
xm = np.zeros((n,1)) # model state

# record the results to calculate and plot.
ym_records = np.zeros(int(simulate_time/dt))
yp_records = np.zeros(int(simulate_time/dt))
e_records = np.zeros(int(simulate_time/dt))

kc = 0 # initial adjustive gain
r = 0.6 # gain
My = 1 # use to plot the y axis

# input signals for the model, a square wave.
# we use a squence for calculation and plot.
yr = r * np.array([np.ones(int(simulate_time/dt/4)), 
                    -np.ones(int(simulate_time/dt/4)),
                    np.ones(int(simulate_time/dt/4)), 
                    -np.ones(int(simulate_time/dt/4))]).flatten()


for i in range(1, len(yr)):
    # yp: plant output
    yp = (Cp @ xp + Dp * u)
    xp_hat = Ap @ xp + Bp * u
    xp = xp + dt * xp_hat
    yp_records[i] = yp

    # ym: model output
    ym = (Cm @ xm + Dm * yr[i-1])
    xk_hat = Am @ xm + Bm * yr[i-1]
    xm = xm + dt * xk_hat
    ym_records[i] = ym

    e_records[i] = ym - yp
    
    # Applying the MIT adaptive law
    kc = kc + dt * gamma * e_records[i-1] * ym_records[i-1]
    u = kc * yr[i]

# display the results
plt.figure()
plt.plot(np.arange(0, simulate_time, dt), yr, label='input signal (yr)', color='black')
plt.plot(np.arange(0, simulate_time, dt), ym_records, label='model output (ym)')
plt.plot(np.arange(0, simulate_time, dt), yp_records, label='plant output (yp)')
plt.axis([0, simulate_time, -My, My])
plt.legend()
plt.show()

