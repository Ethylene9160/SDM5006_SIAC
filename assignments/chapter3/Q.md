# Exercise 2.1

Identify the 2^nd^ order system below using the weighted least squares method.
$$
Y(s)=\frac{2s+3}{s^2+4s+5}U(s)
$$
Where $Y(s)$ and $U(s)$ are the output and input of the system. Let the weighting factor be $\beta(N,k)=0.9^{N-k}$, for $k=1,2,...$



# Exercise 2.2

Let the difference equation of the single-input-single-output system be:
$$
y(k)+a_1y(k-1)+a_2y(k-2)=b_1u(k-1)+b_2u(k-2)+V(k)\\
V(k)=c_1v(k)+c_2v(k-1)+c_3v(k-2)
$$
Take:
$$
a_1=1.6,a_2=0.7,b_1=1.0,b_2=0.4,c_1=0.9,c_2=1.2,c_3=0.3
$$
and the input signal adopts a 4th-order M-sequence with an amplitude of 1.

When the mean and variance of Gaussian noise $v(k)$ are 0 and 0.5, the parameters are estimated by least squares method, recursive least squares method and forgetting-factor
recursive least squares method ($\lambda=0.95$), respectively.

Through the analysis and comparison of the identification results of the three methods, explain the advantages and disadvantages of the above three parameter identification
methods.