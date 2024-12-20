# This is a python implemention for example611.m
import numpy as np

def sindiophantine(
        a:np.array, 
        b:np.array, 
        c:np.array, 
        d:int):
    na:int = len(a)-1
    nb:int = len(b)-1
    nc:int = len(c)-1
    ne:int = d-1
    ng:int = na-1

    ad = np.concatenate((a, np.zeros(ng + ne + 1 - na)))
    cd = np.concatenate((c, np.zeros(ng + d - nc)))

    e = np.zeros(ne+1)
    e[0] = 1

    g = np.zeros(ng+1)

    # Calculate e
    for i in range(1, ne+1):
        e[i] = 0
        for j in range(1, i):
            e[i] += e[i+1-j]*ad[j]
        e[i] = cd[i]-e[i]

    # Calculate g
    for i in range(ng+1):
        g[i] = 0
        for j in range(ne+1):
            g[i] += e[ne+-j] * ad[i+j]
        g[i] = cd[i+d] - g[i]
    f = np.convolve(b, e)
    return e, f, g
        
a = np.array([1, -1.7, 0.7])
b = np.array([1, 0.5])
c = np.array([1, 0.2])
d = 4

e, f, g = sindiophantine(a, b, c, d)

# display
print('e: ', e)
print('f: ', f)
print('g: ', g)


# todo: the results are not correct:
# true values are:
# e = 1.0000    1.9000    2.5300    2.9710
# f = 1.0000    2.4000    3.4800    4.2360    1.4855
# g =  3.2797   -2.0797