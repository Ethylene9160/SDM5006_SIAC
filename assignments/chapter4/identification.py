# this is for assignment 3. which is really hard.
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

# set the font of the figure
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用SimHei字体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# the system to be identification is:
# y(k)+a1y(k-1)+a2y(k-2)=b1u(k-1)+b2u(k-2)+V(k)
# V(k)=c1v(k)+c2v(k-1)+c3v(k-2)
# where v(k) is a white noise with a variance of 0.5. (Gaussian noise with 0 to be mean value)
# where a1=1.6, a2=0.7, b1=1.0, b2=0.4, c1=0.9, c2=1.2, c3=0.3

# time = 1000
# dt = 1
m_seq_order = 4 # order of the m-sequence
m_seq_np = 2**m_seq_order - 1 # period of the m-sequence
# N = int(time/dt)
r = 5
N = r * m_seq_np 
# N = 5

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
m_seq = generate_m_sequence(m_seq_order, N)

# generate v(k)
v = 0.1 * np.random.normal(0, 0.5, N)

# generate y(k)
y = np.zeros(N)
for i in range(2, N):
    y[i] = -1.6 * y[i-1] - 0.7 * y[i-2] + 1.0 * m_seq[i-1] + 0.4 * m_seq[i-2] + \
            0.9 * v[i]+1.2 * v[i-1]+0.3 * v[i-2]

# generate g(k) (impulse response)
# g(k) is used for validation, not for calculation.
g = np.zeros(N)
g[1] = 1.0
g[2] = -1.6 * g[1] - 0.7 * g[0] + 0.4
for i in range(3, N):
    g[i] = -1.6 * g[i-1] - 0.7 * g[i-2]
print(g[:4]) # should be 0, 1, -1.2, 1.22

# least square method, for test.
def LS() -> tuple:
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
    plt.title('Least Square (Maximum Likelihood) Identification')
    plt.show()
    return theta

def CAI() -> tuple:
    # impulse
    u_conv = np.zeros(N)
    u_conv = np.concatenate((np.zeros(N), m_seq))
    g_hat = np.zeros(N)
    for i in range(N):
        g_hat[i] = m_seq_np/(m_seq_np+1) * sum(y[j] * u_conv[N+j-i] for j in range(N-i)) / N
    
    # g_hat = g # for test, to test whether the estimated values are correct.

    # calculate a1, a2, b1, b2.
    HA = np.array([[g_hat[3], g_hat[2]], [g_hat[4], g_hat[3]]])
    theta_a = np.linalg.inv(HA) @ np.array([-g_hat[4], -g_hat[5]])
    a1, a2 = theta_a[0], theta_a[1]

    HB = np.array([[1, 0], [a1, 1]])
    theta_b = HB @ np.array([g_hat[1], g_hat[2]])
    b1, b2 = theta_b[0], theta_b[1]
    theta = np.array([a1, a2, b1, b2])
        
    # result
    y_identified = np.zeros(N)
    for i in range(2, N):
        y_identified[i] = -a1 * y_identified[i-1] - a2 * y_identified[i-2] + b1 * m_seq[i-1] + b2 * m_seq[i-2]

    plt.figure()
    plt.subplot(211)
    plt.plot(y, label='real output y(k)')
    plt.plot(y_identified, label='identified y(k) identified')
    plt.plot(m_seq, label='M-sequence input')
    plt.legend()
    plt.title('Correlation Analysis Identification')
    plt.subplot(212)
    plt.plot(g_hat, label='g_\{hat\}(k)')
    plt.plot(g, label='g(k)')
    plt.legend()
    plt.title('g(k)')
    plt.tight_layout()
    plt.show()
    
    return theta

# recursive gradient method
def RG(alpha = 0.5, c = 1.0, epoch = 1) -> tuple:
    theta = np.random.rand(4)
    for i in range(epoch):
        for j in range(2, N):
            phi = np.array([-y[j-1], -y[j-2], m_seq[j-1], m_seq[j-2]])
            left = alpha/(c+phi.T @ phi) * (y[j]-phi.T@theta) * phi
            theta = theta + left
    # for i in range(epoch):
    #     grad = np.zeros(4)
    #     for j in range(2, N):
    #         grad += (y[j] + theta[0] * y[j-1] + theta[1] * y[j-2] - theta[2] * m_seq[j-1] - theta[3] * m_seq[j-2]) * np.array([-y[j-1], -y[j-2], m_seq[j-1], m_seq[j-2]])
    #     theta -= lr * grad
    #display the result
    y_identified = np.zeros(N)
    for i in range(2, N):
        y_identified[i] = -theta[0] * y_identified[i-1] - theta[1] * y_identified[i-2] + theta[2] * m_seq[i-1] + theta[3] * m_seq[i-2]
    plt.figure()
    plt.plot(y, label='real output y(k)')
    plt.plot(y_identified, label='identified y(k) identified')
    plt.plot(m_seq, label='M-sequence input')
    plt.legend()
    plt.title('Recursive Gradient Identification')
    plt.show()
    return theta

# Stochastic Newton method
def SNM() -> tuple:
    theta = np.zeros((4,1))
    R = np.eye(4)
    for j in range(2, N):
        phi = np.array([-y[j-1], -y[j-2], m_seq[j-1], m_seq[j-2]]).reshape((4,1))
        # phi_mat = phi.reshape(4, 1)
        # print('shape of phi:', phi_mat.shape)
        # tmp = phi_mat @ phi_mat.T
        # print(tmp.shape)
        R = R + (phi @ phi.T - R)/(j-1)
        # R cannot be to small or even singular
        if abs(np.linalg.det(R)) < 1e-7:
            R = np.eye(4)
        theta = theta + np.linalg.inv(R) @ phi * (y[j] - phi.T @ theta)/(j-1)
    theta = theta.reshape(4)

    a1, a2, b1, b2 = theta
    # plot result
    y_identified = np.zeros(N)
    for i in range(2, N):
        y_identified[i] = -a1 * y_identified[i-1] - a2 * y_identified[i-2] + b1 * m_seq[i-1] + b2 * m_seq[i-2]

    plt.figure()
    plt.plot(y, label='real output y(k)')
    plt.plot(y_identified, label='identified y(k) identified')
    plt.plot(m_seq, label='M-sequence input')
    plt.legend()
    plt.title('Recursive Gradient Identification')
    plt.show()
    return theta

def CA() -> tuple:
    ny = 1
    n = m_seq_np*4
    Y1 = y.copy()[ny : n+ny]
    X1 = m_seq.copy()[ny : n+ny]
    # YAvr = np.mean(Y1)
    # Y1 = Y1 - YAvr
    # Rmz = np.zeros(m_seq_np)

    # # calculate correlation matrix, and convolution
    # for i in range(m_seq_np):
    #     sum = 0
    #     for j in range(m_seq_np):
    #         if j >= i:
    #             sum += X1[j-i]*Y1[j]
    #         else:
    #             sum += X1[m_seq_np-i+j]*Y1[j]

    #     Rmz[i] = sum / m_seq_np

    # c = -Rmz[m_seq_np-1]
    # g_hat = 4*(m_seq_np * (Rmz+c))/((m_seq_np+1))+0.2

    
    R_uu= np.zeros(n)
    # R_uu[0] = 1
    # R_uu[else] = -1/m_seq_np
    R_uu[0] = 1.0
    R_uu[1:] = [-1.0/m_seq_np for i in range(1, n)]

    # R_uu = np.correlate(X1, X1, 'full')/n
    for i in range(n):
        for j in range(n):
            if j >= i:
                R_uu[i] += X1[j]*X1[j-i]
            else:
                R_uu[i] += X1[n-i+j]*X1[j]
        R_uu[i] /= n
    print('Ruu: ', R_uu)
    alpha = R_uu[0]
    R_uy = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j >= i:
                R_uy[i] += X1[j]*Y1[j-i]
            else:
                R_uy[i] += X1[n-i+j]*Y1[j]
        R_uy[i] /= n
    c = np.sum(R_uy[:m_seq_np])
    # R_uy = np.correlate(X1, Y1, 'full')/n
    g_hat = m_seq_np * (R_uy + c) / (m_seq_np + 1)

    # calculate a1, a2, b1, b2.
    HA = np.array([[g_hat[3], g_hat[2]], [g_hat[4], g_hat[3]]])
    theta_a = np.linalg.inv(HA) @ np.array([-g_hat[4], -g_hat[5]])
    a1, a2 = theta_a[0], theta_a[1]

    HB = np.array([[1, 0], [a1, 1]])
    theta_b = HB @ np.array([g_hat[1], g_hat[2]])
    b1, b2 = theta_b[0], theta_b[1]
    theta = np.array([a1, a2, b1, b2])

    # plot
    y_identified = np.zeros(N)
    for i in range(2, N):
        y_identified[i] = -a1 * y_identified[i-1] - a2 * y_identified[i-2] + b1 * m_seq[i-1] + b2 * m_seq[i-2]

    plt.figure()
    plt.subplot(211)
    plt.plot(y, label='real output y(k)')
    plt.plot(y_identified, label='identified y(k) identified')
    plt.plot(m_seq, label='M-sequence input')
    plt.legend()
    plt.title('Correlation Analysis Identification')
    plt.subplot(212)
    plt.plot(g_hat, label='g_\{hat\}(k)')
    plt.plot(g, label='g(k)')
    plt.legend()
    plt.title('g(k)')
    plt.tight_layout()
    plt.show()
    
    return theta

# display the result and simulate with the identified model
print('original values: a1=1.6, a2=0.7, b1=1.0, b2=0.4')
# theta_ls = LS()
# print('The identified model is: y(k)+%.2fy(k-1)+%.2fy(k-2)=%.2fu(k-1)+%.2fu(k-2)+noise' % (theta_ls[0], theta_ls[1], theta_ls[2], theta_ls[3]))
theta_cai = CA()
print('The identified model is: a1 = %.2f, a2 = %.2f, b1 = %.2f, b2 = %.2f' % (theta_cai[0], theta_cai[1], theta_cai[2], theta_cai[3]))
# theta_rg = RG()
# print('The identified model is: a1 = %.2f, a2 = %.2f, b1 = %.2f, b2 = %.2f' % (theta_rg[0], theta_rg[1], theta_rg[2], theta_rg[3]))
# theta_snm = SNM()
# print('The identified model is: a1 = %.2f, a2 = %.2f, b1 = %.2f, b2 = %.2f' % (theta_snm[0], theta_snm[1], theta_snm[2], theta_snm[3]))
