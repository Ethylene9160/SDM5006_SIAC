# This is a python version for example211.m
import numpy as np

N = 300 # simulation length

# generate a 4th-order M-sequence with an amplitude of 1
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
m_seq_order = 4
m_seq = generate_m_sequence(m_seq_order, N)

u = m_seq

# generate v(k) - noise
v = 0.1 * np.random.normal(0, 1, N)

# generate y(k)
y = np.zeros(N)
for i in range(1, 5):
    y[i] = \
        1.5 * 0 if i < 1 else y[i-1] - \
        0.7 * 0 if i < 2 else y[i-2] + \
        1.0 * 0 if i < 3 else u[i-3] + \
        0.5 * 0 if i < 4 else u[i-4] + v[i]
for i in range(5, N):
    y[i] = 1.5 * y[i-1] - 0.7 * y[i-2] + 1.0 * u[i-3] + 0.5 * u[i-4] + v[i]

# recursive least square method
theta = np.zeros((4,1)) # [a1, a2, b1, b2].T
K = np.zeros((4,1))
P = 1000000 * np.eye(4)
for i in range(4, N):
    phi = np.array([-y[i-1], -y[i-2], u[i-3], u[i-4]]).reshape((4,1))
    theta = theta + K * (y[i] - phi.T @ theta)
    K = P @ phi / (1 + phi.T @ P @ phi)
    P = P - K @ phi.T @ P
theta = theta.reshape(4)
    
a1, a2, b1, b2 = theta
print('a1 = %.2f, a2 = %.2f, b1 = %.2f, b2 = %.2f' % (a1, a2, b1, b2))
