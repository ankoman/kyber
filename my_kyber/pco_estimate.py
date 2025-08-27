import math
from scipy.stats import norm, entropy
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
    V = 208*3
    M_r = 210
    n_d = 2
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    p_priori = [6/16, 4/16, 1/16]
    print(list_index)

    ### Probabilities
    p_X1_H1 = 0 ### Pr(X=1|H_1)
    for idx in list_index:
        if idx + 2*U + V >= 832:
            p_X1_H1 += prob_U * p_priori[2]
        if idx + 1*U + V >= 832:
            p_X1_H1 += prob_U * p_priori[1]
    p_X0_H1 = 1 - p_X1_H1 ### Pr(X=0|H_1)

    p_X1_H1_ = 0 ### Pr(X=1|H_1_)
    for idx in list_index:
        if idx + -2*U + V >= 832:
            p_X1_H1_ += prob_U * p_priori[2]
        if idx + -1*U + V >= 832:
            p_X1_H1_ += prob_U * p_priori[1]
        if idx + 0*U + V >= 832:
            p_X1_H1_ += prob_U * p_priori[0]
    p_X0_H1_ = 1 - p_X1_H1_ ### Pr(X=0|H_1_)

    p_error = 0
    for idx in list_index:
        if idx + 2*U > 832:
            p_error += prob_U
    p_Y1 = 1 - (1 - p_error)**58 ###  Pr(Y=1)

    p_X0 = p_X0_H1*5/16 + p_X0_H1_*11/16 ### Pr(X=0)
    a = p_X0*p_Y1 ### Pr(Y=1,X=0)
    p_Y1_H1 = a + (1-a)*p_X1_H1 ### Pr(Y=1|H_1)
    p_Y1_H1_ = a + (1-a)*p_X1_H1_ ### Pr(Y=1|H_1_)

    print(p_Y1)
    print(p_X1_H1)
    print(p_X1_H1_)
    print(p_Y1_H1)
    print(p_Y1_H1_)

    ### KL Divergence
    KL_H1 = entropy([p_Y1_H1, 1 - p_Y1_H1], [p_Y1_H1_, 1 - p_Y1_H1_])
    KL_H1_ = entropy([p_Y1_H1_, 1 - p_Y1_H1_], [p_Y1_H1, 1 - p_Y1_H1])
    print(KL_H1, KL_H1_)

    ### Expected number of observations to achieve an LLR (0.95/0.05)
    k_H1 = (math.log2(0.95/0.05) - math.log2(sum(p_priori[1:3])/sum(p_priori)))/KL_H1
    k_H1_ = (math.log2(0.95/0.05) - math.log2(sum(p_priori)/sum(p_priori[1:3])))/KL_H1_
    k = sum(p_priori[1:3]) * k_H1 + sum(p_priori) * k_H1_

    print(k)

def H5_768():
    U = 300
    V = 208*3
    M_r = 210
    n_d = 2
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    p_priori = [6/16, 4/16, 1/16]
    print(list_index)

    ### Probabilities
    p_X1_H5 = 0 ### Pr(X=1|H_5)
    for idx in list_index:
        if idx + 2*U + V >= 832:
            p_X1_H5 += prob_U * p_priori[2]
    p_X0_H5 = 1 - p_X1_H5 ### Pr(X=0|H_5)

    p_X1_H5_ = 0 ### Pr(X=1|H_5_)
    for idx in list_index:
        if idx + -2*U + V >= 832:
            p_X1_H5_ += prob_U * p_priori[2]
        if idx + -1*U + V >= 832:
            p_X1_H5_ += prob_U * p_priori[1]
        if idx + 0*U + V >= 832:
            p_X1_H5_ += prob_U * p_priori[0]
        if idx + 1*U + V >= 832:
            p_X1_H5_ += prob_U * p_priori[1]
    p_X0_H5_ = 1 - p_X1_H5_ ### Pr(X=0|H_5_)

    p_error = 0
    for idx in list_index:
        if idx + 2*U > 832:
            p_error += prob_U
    p_Y1 = 1 - (1 - p_error)**58 ###  Pr(Y=1)

    p_X0 = p_X0_H5*1/16 + p_X0_H5_*15/16 ### Pr(X=0)
    a = p_X0*p_Y1 ### Pr(Y=1,X=0)
    p_Y1_H5 = a + (1-a)*p_X1_H5 ### Pr(Y=1|H_5)
    p_Y1_H5_ = a + (1-a)*p_X1_H5_ ### Pr(Y=1|H_5_)

    print(p_Y1)
    print(p_X1_H5)
    print(p_X1_H5_)
    print(p_Y1_H5)
    print(p_Y1_H5_)

    ### KL Divergence
    KL_H5 = entropy([p_Y1_H5, 1 - p_Y1_H5], [p_Y1_H5_, 1 - p_Y1_H5_])
    KL_H5_ = entropy([p_Y1_H5_, 1 - p_Y1_H5_], [p_Y1_H5, 1 - p_Y1_H5])
    print(KL_H5, KL_H5_)

    ### Expected number of observations to achieve an LLR (0.95/0.05)
    k_H5 = (math.log2(0.95/0.05) - math.log2(1/15))/KL_H5
    k_H5_ = (math.log2(0.95/0.05) - math.log2(15))/KL_H5_
    k = (1/16) * k_H5 + (15/16) * k_H5_

    print(k)

if __name__ == '__main__':
    H5_768()
