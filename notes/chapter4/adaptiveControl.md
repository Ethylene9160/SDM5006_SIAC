

**Definition:** a controller with an *adjustable structure* and/or parameter mechanism.

>  For example: a variable parameter PID controller.

**Form:** the adaptive controller form is a nonlinear or linear time-varying control law.

**Effect**: according to the environment (including noise and interference) and the change of the controlled plant itself (parameter and structure change), adjust the controller structure and/or parameters to reduce or even eliminate these "changes" that affect control performance.

* Compare with robust control: 
  * Robust control: responding to changes with invariance 
  * Adaptive control: variable control

## Types of Adaptive Control

### Self-tuning regulators (STR)

Different design methods + different identification methods=multiple adaptive schemes;

The design is not based on stability considerations and lacks the overall design scheme of "top to bottom".

![image-20241122211856408](/Users/dongjinda/Library/Application Support/typora-user-images/image-20241122211856408.png)

* Dual control



### Model-reference adaptive systems

Key: parameter adjustment mechanism.

- Parameter optimization design method; (MIT scheme)
- Design method based on **Lyapunov** stability theory;
- Design method based on Popov's hyperstability theory.

### Gain scheduling

### Iterative learning control and repetitive control

### Reinforcement learning adaptive control

