
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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


# 仿真参数
kp, ki, kd =0.8, 0.2, 0.05  # PID参数
pid = PIDController(kp, ki, kd)  # 初始化PID控制器
# nnpid = NNPID([3, 32, 3], 0.0001)
# single_nnpid = SingleNNPID(0.001, 0.001, 0.001,
#                            kp_factor = 1,
#                            ki_factor = 1,
#                            kd_factor = 1)
setpoint = 0.6  # 目标值
setpoints = []
dt = 0.05  # 时间步长
sim_time = 100  # 仿真总时间
steps = int(sim_time / dt)  # 仿真步数

# 初始化系统状态
x1 = 0  # 初始输出
x2 = 0  # 初始输出导数
ref_x1 = 0
ref_x2 = 0
outputs = []  # 系统输出存储
ref_outputs = []
control_signals = []  # 控制信号存储
ref_control_signals = []
time = np.arange(0, sim_time, dt)  # 时间序列
errors = error_store()
ref_errors = error_store()

km = 1
kc = 0
My = 1

gamma = 0.1
err = 0

u_ref = 0
# 仿真主循环
for t in time:
    if (t == 25 or t == 75):
        setpoint = -0.6
    elif (t == 50 ):
        setpoint = 0.6
    setpoints.append(setpoint)
    # 计算误差
    error = setpoint - x1
    ref_error = setpoint - ref_x1

    # applying MIT law
    kc_temp = kc+dt*gamma*err

    current_errors = errors.update(error, dt)
    ref_current_errors = ref_errors.update(ref_error, dt)
    # 使用PID控制器计算控制信号
    # u = pid.cal_u(error, dt)
    u = pid.get_u(current_errors)
    control_signals.append(u)
    
    # training online
    # print('output shape: ', np.array(nn_current_errors).shape)
    # print('output shape: ', nnpid.predict(nn_current_errors).shape)
    # nn_ks = nnpid.predict(nn_current_errors)
    # nn_ks = single_nnpid.K * single_nnpid.factors
    # nn_p.append(nn_ks[0])
    # nn_i.append(nn_ks[1])
    # nn_d.append(nn_ks[2])
    # nn_u = np.dot(nn_ks, nn_current_errors)
    # nn_control_signals.append(nn_u)


    # 更新系统状态（离散化状态方程），不用管
    dx1 = x2
    # dnn_x1 = nn_x2
    dref_x1 = ref_x2

    dx2 = -1 * x2 - x1 + u
    dref_x2 = -1 * ref_x2 - ref_x1 + kc_temp * error

    x1 += dx1 * dt
    x2 += dx2 * dt

    ref_x1 += dref_x1 * dt
    ref_x2 += dref_x2 * dt

    # nn_x1 += dnn_x1 * dt
    # nn_x2 += dnn_x2 * dt
    # 系统状态更新结束

    # 训练神经网络
    # nnpid.onlineTraining(nn_current_errors, nn_x1, nn_current_errors, setpoint)
    # single_nnpid.onlineTraining(nn_current_errors, nn_x1)
    # 记录输出
    outputs.append(x1)

    err = ref_x1 - x1
    kc = kc_temp
    # nn_outputs.append(nn_x1)


# 可视化结果
plt.figure(figsize=(10, 6))

# 输出响应
plt.subplot(3, 1, 1)
plt.plot(time, outputs, label="PID Output (y)")
# plt.plot(time, nn_outputs, label="NN Output (y)")
# plt.axhline(setpoint, color='r', linestyle='--', label="Setpoint")
plt.plot(time, setpoints, label="Setpoint", linestyle = '--', color='r')
plt.title("System Output vs Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
# control_signals[0] = 0
# nn_control_signals[0] = 0
# 控制信号
plt.subplot(3, 1, 2)
plt.plot(time[1:], control_signals[1:], label="Control Signal (u)", color='g')
# plt.plot(time[1:], nn_control_signals[1:], label="NN Control Signal (u)", color='b')
plt.title("Control Signal")
plt.xlabel("Time (s)")
plt.ylabel("Control Signal")
plt.legend()
# nn_p[0] = 0
# nn_i[0] = 0
# nn_d[0] = 0
# p i d values
# plt.subplot(3, 1, 3)
# plt.plot(time[1:], nn_p[1:], label="NN P")
# plt.plot(time[1:], nn_i[1:], label="NN I")
# plt.plot(time[1:], nn_d[1:], label="NN D")
# plt.title("NN PID values")
# plt.xlabel("Time (s)")
# plt.ylabel("PID values")
# plt.legend()

plt.tight_layout()
plt.show()

