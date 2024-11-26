import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class NNPID(torch.nn.Module):
    def __init__(self, layer_sizes, lr):
        super(NNPID, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # tensor 0
        self.last_y = torch.tensor([0], dtype=torch.float32)
        self.last_u = torch.tensor([0], dtype=torch.float32)
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        x = x #* torch.tensor([5, 1, 0.5], dtype=torch.float32)
        return x

    def onlineTraining(self, inputs, realOutputs, errors, setpoint):
        # convert inputs, realOutputs, errors to torch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        errors = torch.tensor(errors, dtype=torch.float32)
        realOutputs = torch.tensor(realOutputs, dtype=torch.float32)
        setpoint = torch.tensor(setpoint, dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self(inputs)

        # Ensure outputs and errors are compatible for dot product
        if outputs.shape != errors.shape:
            print(f"Shape mismatch: outputs.shape={outputs.shape}, errors.shape={errors.shape}")
            return

        u = torch.matmul(outputs, errors).view(1, 1)  # Convert u to shape (1, 1)
        sgn = 1 if (self.last_u - u) * (self.last_y - realOutputs) >= 0 else -1

        # error is scalar
        error = (realOutputs-setpoint) * sgn
        # clip the error
        error = torch.clamp(error, -0.5, 0.5).view(1,1)

        # print('error: ', error)
        # print('u: ', u)

        # back propagate the error
        # u.grad.data.clamp_(-1, 1)
        u.backward(error)  # Scalar error
        self.optimizer.step()

        self.last_u = u
        self.last_y = realOutputs

    def predict(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        return self(inputs).detach().numpy()

    # def cal_u(self, errors):
    #     return self.predict(errors)

class SingleNNPID:
    def __init__(self,
                 lr_p,
                 lr_i,
                 lr_d,
                 kp_factor:float = 1,
                 ki_factor:float = 1,
                 kd_factor:float = 1):
        # self.lr_p = lr_p
        # self.lr_i = lr_i
        # self.lr_d = lr_d
        self.lr = np.array([lr_p, lr_i, lr_d])
        self.K = np.array([0.5, 0.1, 0.05])
        self.factors = np.array([kp_factor, ki_factor, kd_factor])
    def onlineTraining(self, errors, realOutput):
        errors = errors.copy()
        errors = np.clip(errors, -0.5, 0.5)
        err = errors[0]
        u = self.get_u(errors)
        self.K[0] += err * self.lr[0] * errors[0] * u
        self.K[1] += err * self.lr[1] * errors[1] * u
        self.K[2] += err * self.lr[2] * errors[2] * u

    def get_u(self, errors):
        k_tmp = self.K * self.factors
        return k_tmp.dot(errors)

    def __norm(self):
        self.K = np.linalg.norm(self.K)

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
nnpid = NNPID([3, 32, 3], 0.0001)
# single_nnpid = SingleNNPID(0.001, 0.001, 0.001,
#                            kp_factor = 1,
#                            ki_factor = 1,
#                            kd_factor = 1)
setpoint = 5  # 目标值
dt = 0.01  # 时间步长
sim_time = 40  # 仿真总时间
steps = int(sim_time / dt)  # 仿真步数

# 初始化系统状态
x1 = 0  # 初始输出
x2 = 0  # 初始输出导数
nn_x1 = 0
nn_x2 = 0
outputs = []  # 系统输出存储
nn_outputs = []
control_signals = []  # 控制信号存储
nn_control_signals = []
time = np.arange(0, sim_time, dt)  # 时间序列
errors = error_store()
nn_errors = error_store()
nn_p = []
nn_i = []
nn_d = []
# 仿真主循环
for t in time:
    # 计算误差
    error = setpoint - x1
    nn_error = setpoint - nn_x1

    current_errors = errors.update(error, dt)
    nn_current_errors = np.array(nn_errors.update(nn_error, dt))
    # 使用PID控制器计算控制信号
    # u = pid.cal_u(error, dt)
    u = pid.get_u(current_errors)
    control_signals.append(u)
    # training online
    # print('output shape: ', np.array(nn_current_errors).shape)
    # print('output shape: ', nnpid.predict(nn_current_errors).shape)
    nn_ks = nnpid.predict(nn_current_errors)
    # nn_ks = single_nnpid.K * single_nnpid.factors
    nn_p.append(nn_ks[0])
    nn_i.append(nn_ks[1])
    nn_d.append(nn_ks[2])
    nn_u = np.dot(nn_ks, nn_current_errors)
    nn_control_signals.append(nn_u)


    # 更新系统状态（离散化状态方程），不用管
    dx1 = x2
    dnn_x1 = nn_x2

    if t > 20: # 不稳定系统
        dx2 = -0.21 * x2 - x1 + 1.5 * u
        dnn_x2 = -0.21 * nn_x2 - nn_x1 + 1.5 * nn_u
    else: # 稳定系统
        dx2 = -2.3 * x2 - x1 + 1.7 * u
        dnn_x2 = -2.3 * nn_x2 - nn_x1 + 1.7 * nn_u

    x1 += dx1 * dt
    x2 += dx2 * dt

    nn_x1 += dnn_x1 * dt
    nn_x2 += dnn_x2 * dt
    # 系统状态更新结束

    # 训练神经网络
    nnpid.onlineTraining(nn_current_errors, nn_x1, nn_current_errors, setpoint)
    # single_nnpid.onlineTraining(nn_current_errors, nn_x1)
    # 记录输出
    outputs.append(x1)
    nn_outputs.append(nn_x1)


# 可视化结果
plt.figure(figsize=(10, 6))

# 输出响应
plt.subplot(3, 1, 1)
plt.plot(time, outputs, label="PID Output (y)")
plt.plot(time, nn_outputs, label="NN Output (y)")
plt.axhline(setpoint, color='r', linestyle='--', label="Setpoint")
plt.title("System Output vs Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend()
control_signals[0] = 0
nn_control_signals[0] = 0
# 控制信号
plt.subplot(3, 1, 2)
plt.plot(time[1:], control_signals[1:], label="Control Signal (u)", color='g')
plt.plot(time[1:], nn_control_signals[1:], label="NN Control Signal (u)", color='b')
plt.title("Control Signal")
plt.xlabel("Time (s)")
plt.ylabel("Control Signal")
plt.legend()
nn_p[0] = 0
nn_i[0] = 0
nn_d[0] = 0
# p i d values
plt.subplot(3, 1, 3)
plt.plot(time[1:], nn_p[1:], label="NN P")
plt.plot(time[1:], nn_i[1:], label="NN I")
plt.plot(time[1:], nn_d[1:], label="NN D")
plt.title("NN PID values")
plt.xlabel("Time (s)")
plt.ylabel("PID values")
plt.legend()

plt.tight_layout()
plt.show()

