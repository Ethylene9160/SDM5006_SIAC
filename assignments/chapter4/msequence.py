# the system to be identification is:
# y(k)+a1y(k-1)+a2y(k-2)=b1u(k-1)+b2u(k-2)
# where a1=1.6, a2=0.7, b1=1.0, b2=0.4
import numpy as np
import matplotlib.pyplot as plt

m_seq_order = 4
m_seq_np = 2**m_seq_order - 1
N = 10 * m_seq_np

# generate M sequence
def generate_m_sequence(order, N):
    # Initialize the shift register with a non-zero state (e.g., [1, 0, 0, 0])
    register = [1] + [0] * (order - 1)
    sequence = []

    # Feedback polynomial: x^4 + x + 1
    feedback_taps = [0, 3]  # The taps correspond to x^4 and x

    # Generate the sequence of length N
    for _ in range(N):
        # Output the current state
        sequence.append(register[0])

        # Compute feedback value based on taps
        feedback = sum(register[tap] for tap in feedback_taps) % 2

        # Shift the register and insert feedback value
        register = [feedback] + register[:-1]

    return sequence
m_seq = generate_m_sequence(m_seq_order, N)

# generate impulse response g(k)
# 输入脉冲信号 u(k)
u = np.zeros(N)
u[0] = 1  # 单位脉冲输入

# 初始化输出 y(k)，即脉冲响应 g(k)
g = np.zeros(N)

# 递推计算脉冲响应
for k in range(2, N):  # 从k=2开始，因为涉及y(k-1)和y(k-2)
    g[k] = -1.6 * g[k-1] - 0.7 * g[k-2] + 1.0 * u[k-1] + 0.4 * u[k-2]

# 画图
plt.figure()
plt.plot(g, label='g(k)')
plt.xlabel('k')
plt.ylabel('g(k)')
plt.title('Impulse Response')
plt.legend()
plt.show()

