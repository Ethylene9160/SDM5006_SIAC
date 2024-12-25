import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

num = [5]
den = [2, 1]
delay = 1  # system delay is 1 second

# Create the transfer function without delay
sys = signal.TransferFunction(num, den)

# Generate the time vector
t = np.linspace(0, 20, num=1000)

# Compute the step response of the system without delay
t_no_delay, y_no_delay = signal.step(sys, T=t)

# Apply the delay by shifting the response
t_with_delay = t + delay
y_with_delay = np.concatenate((np.zeros(int(delay * (len(t) / t[-1]))), y_no_delay))[:len(t)]

# Plot the step response with delay
plt.figure()
plt.plot(t, y_with_delay)
plt.title('Step Response of the System')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.xlim([0, 20])
plt.grid()
plt.show()