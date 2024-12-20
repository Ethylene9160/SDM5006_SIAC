# This is a python 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# enable latex
# plt.rcParams['text.usetex'] = True

# PID controller
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

num = np.array([1])
den = np.array([1, 3, 3, 1])

plant = signal.TransferFunction(num, den)
Ap, Bp, Cp, Dp = signal.tf2ss(plant.num, plant.den)

# display Ap, Bp:
print("Ap: ", Ap)
print("Bp: ", Bp)

n = 3
m = 1

simulation_time = 100
dt = 0.1

x = np.zeros((n,1))
u = 0

x_records = [x]
y_records = [0]
setpoint = 0.2
controller = PIDController(0.15,1,0)
for i in range(int(40/dt)):
    x_dot = Ap @ x + Bp * u
    y = Cp@x
    x = x + dt * x_dot
    # print('shape of y: ', y.shape)

    x_records.append(x)
    y_records.append(float(y))

    u = controller.cal_u(setpoint - y, dt)
    u = u**4
    # print('u: ', u)

plt.figure()
plt.plot(y_records)
# plt.plot()
plt.show()



