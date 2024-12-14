# this is for assignment 3. which is really hard.
import numpy as np
import scipy
import matplotlib.pyplot as plt

# the system to be identification is:
# y(k)+a1y(k-1)+a2y(k-2)=b1u(k-1)+b2u(k-2)+V(k)
# V(k)=c1v(k)+c2v(k-1)+c3v(k-2)
# where v(k) is a white noise with a variance of 0.5. (Gaussian noise with 0 to be mean value)
# where a1=1.6, a2=0.7, b1=1.0, b2=0.4, c1=0.9, c2=1.2, c3=0.3

time = 100
dt = 0.1
N = int(time/dt)

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
m_seq = generate_m_sequence(4, N)

# generate v(k)
# 暂时不加噪声，不然真不知道是算错了还是噪声导致的（）
v = 0.0 * np.random.normal(0, 0.5, N)

# least square method, for test.
def LS() -> tuple:
    # generate y(k)
    y = np.zeros(N)
    for i in range(2, N):
        y[i] = -1.6 * y[i-1] - 0.7 * y[i-2] + 1.0 * m_seq[i-1] + 0.4 * m_seq[i-2] + \
                0.9 * v[i]+1.2 * v[i-1]+0.3 * v[i-2]
    
    # generate the matrix
    Y = np.zeros((N-2, 4))
    for i in range(2, N):
        Y[i-2] = np.array([-y[i-1], -y[i-2], m_seq[i-1], m_seq[i-2]])
    
    # calculate the coefficients
    theta = np.linalg.inv(Y.T @ Y) @ Y.T @ y[2:]

    plt.figure()
    plt.plot(y[2:], label='y(k)')
    plt.plot(Y @ theta, label='y(k) identified')
    plt.plot(m_seq, label='m-sequence')
    plt.legend()
    plt.show()
    return theta

# display the result and simulate with the identified model
theta = LS()
print('The identified model is: y(k)+%.2fy(k-1)+%.2fy(k-2)=%.2fu(k-1)+%.2fu(k-2)+noise' % (theta[0], theta[1], theta[2], theta[3]))


