# This is a python implimentation for example162.m,
# generating a 4-order m-sequence.

import numpy as np
import matplotlib.pyplot as plt

n = 4 # number of registers
m = 2**n - 1 # length of m-sequence
reg = np.ones(n, dtype=int) # initial value of registers: [A1, A2, A3, A4] = [1, 1, 1, 1]
c = np.array([0, 0, 1, 1]) # [c1, c2, c3, c4] = [0, 0, 1, 1]
seq = np.zeros(m, dtype=int) # initial value of m-sequence

for i in range(m):
    seq[i] = reg[-1] # y(k) = A4(k)
    a1_new = c[0] * reg[0] ^ c[1] * reg[1] ^ c[2] * reg[2] ^ c[3] * reg[3]
    # A4 = A3, A3 = A2, A2 = A1, A1 = new
    reg[3] = reg[2]
    reg[2] = reg[1]
    reg[1] = reg[0]
    reg[0] = a1_new # update the first register

plt.figure()
# 黑色线条
plt.stairs(seq, color='k')
# y轴间隔0.5
plt.yticks(np.arange(0, 2, 1))
plt.ylim([-0.5, 1.5])
plt.xlim([0, m])
plt.xticks(np.arange(0, m+1, 1))
plt.show()