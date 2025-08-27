import math
from scipy.stats import norm
from decimal import *
import numpy as np

def number_of_2():
    a = 59
    p = 1 - norm.cdf((a-0.5-96)/9.16, 0, 1)
    print(f'{p=}, {p > 1-2**-16}')

def delta_w():
    dv = 4
    i_to = 3328 // (2**(dv+1))
    sigma = math.sqrt(236*2)
    th = -50

    sum = 0
    for i in range(-i_to, i_to+1):
        sum += norm.cdf(th, i, sigma)

    sum *= 2**dv
    sum /= 3328 + 2**dv
    print(sum)

def H1_768():
    U = 300
    V = 208*1
    M_r = 210
    n_d = 2
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    p_priori = [6/16, 4/16, 1/16]

    ### Probabilities
    p_X1_H1 = 0 ### Pr(X=1|H_1)
    for idx in list_index:
        if idx + 2*U > V:
            p_X1_H1 += prob_U * p_priori[2]
        if idx + 1*U > V:
            p_X1_H1 += prob_U * p_priori[1]

    p_X0_H1 = 1 - p_X1_H1 ### Pr(X=0|H_1)

    p_X1_H1_ = 0 ### Pr(X=1|H_1_)
    for idx in list_index:
        if idx + -2*U > V:
            p_X1_H1_ += prob_U * p_priori[2]
        if idx + -1*U > V:
            p_X1_H1_ += prob_U * p_priori[1]
        if idx + 0*U > V:
            p_X1_H1_ += prob_U * p_priori[0]

    p_X0_H1_ = 1 - p_X1_H1_ ### Pr(X=0|H_1_)
    print(p_X1_H1)
    print(p_X1_H1_)

if __name__ == '__main__':
    H1_768()
