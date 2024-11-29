

## Adaptive Control Based on the Gradient Method

### MIT adaptive law

> Notation:
>
> |          |      |                                                              |
> | -------- | ---- | ------------------------------------------------------------ |
> | $y_m(t)$ |      | Output through the reference model.                          |
> | $y_p(t)$ |      | System real output.                                          |
> | $\theta$ |      | the unknown or slow time-varying parameter of the controller of the plant. |

Target: $\min\{J(\theta)=\frac12e^2(t)\}$

Now, let:
$$
\frac{d\theta(t)}{dt}=-\gamma\frac{\partial J(\theta)}{\partial \theta}=-\gamma e(t)\frac{\partial e(t)}{\partial \theta}
$$
We define $\frac{\partial e(t)}{\partial \theta}$ to be **sensitivity derivative**. We call this law to be the **MIT** law.

* Apply MIT model on adjustable gain MRAC

For a system with known model whose real transger function is $k_pG(s)$, where $k_p$ is unknown or slowly time-varying, but $G(s)$ is known which is stable and minimum phase.

Let the *regference model* to be $k_mG(s)$ where $k_m$ is the reference model gain.

![image-20241129200928198](/Users/dongjinda/Library/Application Support/typora-user-images/image-20241129200928198.png)

The tracking error:
$$
e(t)=y_m(t)-y_p(t)=k_mG(p)y_r(t)-k_pG(p)u(t)
$$

> p: differential operator, $p=\frac{d}{dt}$

Here,
$$
y_p(t)&=&k_pG(p)u(t)\\
y_m(t)&=&k_mG(p)y_r(t)\\
u(t)&=&\theta(t)y_r(t)\\
$$
Because only in term $u(t)$ contains $\theta$, so the sensitivity derivative will be:
$$
\frac{\partial e(t)}{\partial \theta}=-k_pG(p)y_r(t)=-\frac {k_p}{k_m}y_m(t)
$$
Hence, the MIT adaptive control rule is:
$$
\frac{d\theta(t)}{dt}&=&-\gamma_pe(t)\frac{\partial e(t)}{\partial \theta}\\
&=&\gamma_p\frac{k_p}{k_m}e(t)y_m(t)\\
&=&\gamma e(t)y_m(t)
$$
Where the adaptive gain:
$$
\gamma=\gamma_m\frac{k_p}{k_m}
$$
With:
$$
u(t)=\theta(t)y_r(t)\\
\theta(t)=\gamma\int e(t)y_m(t)dt
$$
