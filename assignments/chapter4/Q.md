# Esercise 3.1

Let the difference equation of the single-input-single-output system be
$$
y(k)+a_1y(k-1)+a_2y(k-2)=b_1u(k-1)+b_2u(k-2)+V(k)\\
V(k)=c_1v(k)+c_2v(k-1)+c_3v(k-2)
$$

Take
$$
a_1=1.6,~a_2=0.7,~b_1=1.0,~b_2=0.4,~c_1=0.9,~c_2=1.2,~c_3=0.3
$$

and the input signal adopts a 4th-order M-sequence with an amplitude of 1.

When the mean and variance of Gaussian noise $v(k)$ are 0 and 0.5, the parameters are
estimated by the maximum likelihood identification method, correlation analysis method, recursive gradient correction estimation method, and recursive stochastic Newton
method, respectively.