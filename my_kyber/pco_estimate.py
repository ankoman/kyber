import math
from scipy.stats import norm, entropy
from decimal import *
import numpy as np

list_U10 = [3,7,10,13,16,20,23,26,29,33,36,39,42,46,49,52,55,59,62,65,68,72,75,78,81,85,88,91,94,98,101,104,107,111,114,117,120,124,127,130,133,137,140,143,146,150,153,156,159,163,166,169,172,176,179,182,185,189,192,195,198,202,205,208,211,215,218,221,224,228,231,234,237,241,244,247,250,254,257,260,263,267,270,273,276,280,283,286,289,293,296,299,302,306,309,312,315,319,322,325,328,332,335,338,341,345,348,351,354,358,361,364,367,371,374,377,380,384,387,390,393,397,400,403,406,410,413,416]

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

def decode(val):
    val = val % 3329
    if val >= 832 and val < 2496:
        return True
    else:
        return False

def expected_num_obs(p, expected_llr):
    ### Expected number of observations until 0/1 is observed to achieve an expected LLR
    q = 0
    expected_num = 0
    for i in range(1, 10000):
        p_i = p * (1-p)**(i-1)
        expected_num += p_i * i
        q += p_i
        llr = math.log2(q/(1-q))
        if llr >= expected_llr:
            break
    return expected_num


def Hi_768(U, V, M_r, n_d, hyp, hyp_, DEBUG=False):
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    p_priori = np.array([6/16, 4/16, 1/16, 1/16, 4/16])   ### -1 = 4/16, -2 = 1/16
    mask = np.zeros_like(p_priori)
    mask[hyp + hyp_] = p_priori[hyp + hyp_]
    p_priori = mask / sum(mask)
    # print(p_priori)
    # print(list_index)

    ### Probabilities
    p_X1_Hi = 0 ### Pr(X=1|H_i)
    p_Hi = sum(p_priori[i] for i in hyp)
    for idx in list_index:
        for s in hyp:
            if decode(idx + s*U + V):
                p_X1_Hi += prob_U * p_priori[s] / p_Hi
    if p_X1_Hi > 1:
        p_X1_Hi = 1
    p_X0_Hi = 1 - p_X1_Hi ### Pr(X=0|H_i)

    p_X1_Hi_ = 0 ### Pr(X=1|H_i_)
    p_Hi_ = sum(p_priori[i] for i in hyp_)
    for idx in list_index:
        for s in hyp_:
            if decode(idx + s*U + V):
                p_X1_Hi_ += prob_U * p_priori[s] / p_Hi_
    if p_X1_Hi_ > 1:
        p_X1_Hi_ = 1
    p_X0_Hi_ = 1 - p_X1_Hi_ ### Pr(X=0|H_i_)

    p_error = 0
    for idx in list_index:
        if decode(idx + 2*U):
            p_error += prob_U
    p_Y1 = 1 - (1 - p_error)**58 ###  Pr(Y=1)

    p_X0 = p_X0_Hi*p_Hi + p_X0_Hi_*p_Hi_ ### Pr(X=0)
    a = p_Y1 ### Pr(Y=1|X=0), X and Y are independent.
    p_Y1_Hi = a + (1-a)*p_X1_Hi ### Pr(Y=1|H_i)
    p_Y1_Hi_ = a + (1-a)*p_X1_Hi_ ### Pr(Y=1|H_i_)

    ### KL Divergence
    KL_Hi = entropy([p_Y1_Hi, 1 - p_Y1_Hi], [p_Y1_Hi_, 1 - p_Y1_Hi_], base=2)
    KL_Hi_ = entropy([p_Y1_Hi_, 1 - p_Y1_Hi_], [p_Y1_Hi, 1 - p_Y1_Hi], base=2)

    ### Expected number of observations to achieve an LLR (0.99/0.01)
    llr = math.log2(0.99/0.01)
    priori_odds = 0#math.log2(p_Hi/p_Hi_)
    priori_odds_ = 0#math.log2(p_Hi_/p_Hi)

    ### Inf handling
    if math.isinf(KL_Hi) and math.isinf(KL_Hi_):
        k_Hi = 1
        k_Hi_ = 1
    else:
        tmp = -math.log2(1+2**llr)
        if math.isinf(KL_Hi):   # An element of [p_Y1_Hi_, 1 - p_Y1_Hi_] is 0
            if p_Y1_Hi_ == 0:   # When Y=1 is observed, H_i is surely true.
                k_Hi = expected_num_obs(p_Y1_Hi, llr) #tmp/math.log2(p_Y1_Hi)
            else:               # When Y=0 is observed, H_i is surely true.
                k_Hi = expected_num_obs(1-p_Y1_Hi, llr) #tmp/math.log2(1-p_Y1_Hi)
        else:
            k_Hi = (llr - priori_odds)/KL_Hi
        if math.isinf(KL_Hi_):  # An element of [p_Y1_Hi, 1 - p_Y1_Hi] is 0
            if p_Y1_Hi == 0:    # When Y=1 is observed, H_i_ is surely true.
                k_Hi_ = expected_num_obs(p_Y1_Hi_, llr) #tmp/math.log2(p_Y1_Hi_)
            else:
                k_Hi_ = expected_num_obs(1 - p_Y1_Hi_, llr) #tmp/math.log2(1-p_Y1_Hi_)
        else:
            k_Hi_ = (llr - priori_odds_)/KL_Hi_

    k = p_Hi * k_Hi + p_Hi_ * k_Hi_

    if DEBUG:
        print(p_X0)
        print(p_Y1)
        print(p_X1_Hi)
        print(p_X1_Hi_)
        print(p_Y1_Hi)
        print(p_Y1_Hi_)

        print(f'Traget LLR: {llr}')
        print(f'{priori_odds=:f}, {priori_odds_=:f}')
        print(f'{KL_Hi=:f}, {KL_Hi_=:f}')

        print(k_Hi, k_Hi_, k)
    return k

def main(hyp, hyp_, M_r, n_d):

    list_min_k = []
    for M_r in [256]:
        for i in range(1, M_r.bit_length()):
            n_d = 2**i
            list_k = []
            for U in list_U10:
                for V in [0, 208, 416, 624, 832, 1040, 1248, 1456]:
                    k = Hi_768(U, V, M_r, n_d, hyp, hyp_)
                    list_k.append(k)
                    print(f'{M_r=}, {n_d=}, {U=}, {V=}, {k=}')
            list_min_k.append(min(list_k))
    print(max(list_min_k))
    print(list_min_k)

if __name__ == '__main__':
    hyp = [1]
    hyp_ = [2]
    U = 260
    V = 416
    M_r = 100
    n_d = 2
    k = Hi_768(U, V, M_r, n_d, hyp, hyp_, True)

    main(hyp, hyp_, M_r, n_d)
