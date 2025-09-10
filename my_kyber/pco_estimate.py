import math
from scipy.stats import norm, entropy
from decimal import *
import numpy as np

list_U10 = [3,7,10,13,16,20,23,26,29,33,36,39,42,46,49,52,55,59,62,65,68,72,75,78,81,85,88,91,94,98,101,104,107,111,114,117,120,124,127,130,133,137,140,143,146,150,153,156,159,163,166,169,172,176,179,182,185,189,192,195,198,202,205,208,211,215,218,221,224,228,231,234,237,241,244,247,250,254,257,260,263,267,270,273,276,280,283,286,289,293,296,299,302,306,309,312,315,319,322,325,328,332,335,338,341,345,348,351,354,358,361,364,367,371,374,377,380,384,387,390,393,397,400,403,406,410,413,416]
list_V4 = [0, 208, 416, 624, 832, 1040, 1248, 1456, 1665, 1873, 2081, 2289, 2497, 2705, 2913, 3121]
list_U11 = [2, 3, 5, 7, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 29, 31, 33, 34, 36, 37, 39, 41, 42, 44, 46, 47, 49, 50, 52, 54, 55, 57, 59, 60, 62, 63, 65, 67, 68, 70, 72, 73, 75, 76, 78, 80, 81, 83, 85, 86, 88, 89, 91, 93, 94, 96, 98, 99, 101, 102, 104, 106, 107, 109, 111, 112, 114, 115, 117, 119, 120, 122, 124, 125, 127, 128, 130, 132, 133, 135, 137, 138, 140, 141, 143, 145, 146, 148, 150, 151, 153, 154, 156, 158, 159, 161, 163, 164, 166, 167, 169, 171, 172, 174, 176, 177, 179, 180, 182, 184, 185, 187, 189, 190, 192, 193, 195, 197, 198, 200, 202, 203, 205, 206, 208, 210, 211, 213, 215, 216, 218, 219, 221, 223, 224, 226, 228, 229, 231, 232, 234, 236, 237, 239, 241, 242, 244, 245, 247, 249, 250, 252, 254, 255, 257, 258, 260, 262, 263, 265, 267, 268, 270, 271, 273, 275, 276, 278, 280, 281, 283, 284, 286, 288, 289, 291, 293, 294, 296, 297, 299, 301, 302, 304, 306, 307, 309, 310, 312, 314, 315, 317, 319, 320, 322, 323, 325, 327, 328, 330, 332, 333, 335, 336, 338, 340, 341, 343, 345, 346, 348, 349, 351, 353, 354, 356, 358, 359, 361, 362, 364, 366, 367, 369, 371, 372, 374, 375, 377, 379, 380, 382, 384, 385, 387, 388, 390, 392, 393, 395, 397, 398, 400, 401, 403, 405, 406, 408, 410, 411, 413, 414, 416]
list_V5 = [0, 104, 208, 312, 416, 520, 624, 728, 832, 936, 1040, 1144, 1248, 1352, 1456, 1560, 1665, 1769, 1873, 1977, 2081, 2185, 2289, 2393, 2497, 2601, 2705, 2809, 2913, 3017, 3121, 3225]

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

def expected_num_obs(p, expected_p):
    ### Expected number of observations until 0/1 is observed to achieve an expected probability
    # q = 0
    # expected_num = 0
    # for i in range(1, 10000):
    #     p_i = p * (1-p)**(i-1)
    #     expected_num += p_i * i
    #     q += p_i
    #     llr = math.log2(q/(1-q))
    #     if q >= expected_p:
    #         break

    if p != 0:
        return 1/p
    else:
        return float('inf') 

def gen_N_Hi(U, V, M_r, n_d, hyp, hyp_, mlkem, DEBUG=False):
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    if mlkem == 512:
        p_priori = np.array([20/16, 15/64, 6/64, 1/64, 1/64, 6/64, 15/64])   ### -1 = 15/64, -2 = 6/64, -3 = 1/64
    else:
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
            if decode(idx - s*U + V):
                p_X1_Hi += prob_U * p_priori[s] / p_Hi
    if p_X1_Hi > 1:
        p_X1_Hi = 1
    p_X0_Hi = 1 - p_X1_Hi ### Pr(X=0|H_i)

    p_X1_Hi_ = 0 ### Pr(X=1|H_i_)
    p_Hi_ = sum(p_priori[i] for i in hyp_)
    for idx in list_index:
        for s in hyp_:
            if decode(idx - s*U + V):
                p_X1_Hi_ += prob_U * p_priori[s] / p_Hi_
    if p_X1_Hi_ > 1:
        p_X1_Hi_ = 1
    p_X0_Hi_ = 1 - p_X1_Hi_ ### Pr(X=0|H_i_)

    beta = 0
    if mlkem == 512:
        e1_max = 3
    else:
        e1_max = 2
    for idx in list_index:
        if decode(idx + e1_max*U):
            beta += prob_U
    alpha = 1 - (1 - beta)**32 ###  Pr(Y=1)

    p_X0 = p_X0_Hi*p_Hi + p_X0_Hi_*p_Hi_ ### Pr(X=0)
    p_Y1_Hi = alpha + (1-alpha)*p_X1_Hi ### Pr(Y=1|H_i)
    p_Y1_Hi_ = alpha + (1-alpha)*p_X1_Hi_ ### Pr(Y=1|H_i_)

    ### KL Divergence
    KL_Hi = entropy([p_Y1_Hi, 1 - p_Y1_Hi], [p_Y1_Hi_, 1 - p_Y1_Hi_], base=2)
    KL_Hi_ = entropy([p_Y1_Hi_, 1 - p_Y1_Hi_], [p_Y1_Hi, 1 - p_Y1_Hi], base=2)

    ### Expected number of observations to achieve an LLR (0.99/0.01)
    target_p = 0.99
    target_llr = math.log2(target_p/(1-target_p))
    priori_odds = math.log2(p_Hi/p_Hi_)
    priori_odds_ = math.log2(p_Hi_/p_Hi)

    ### Inf and minus handling
    KL_Hi = abs(KL_Hi)
    KL_Hi_ = abs(KL_Hi_)
    if math.isinf(KL_Hi) and math.isinf(KL_Hi_):
        N_Hi = 1
        N_Hi_ = 1
    else:
        tmp = -math.log2(1+2**target_llr)
        if math.isinf(KL_Hi):   # An element of [p_Y1_Hi_, 1 - p_Y1_Hi_] is 0
            if p_Y1_Hi_ == 0:   # When Y=1 is observed, H_i is surely true.
                N_Hi = expected_num_obs(p_Y1_Hi, target_p) #tmp/math.log2(p_Y1_Hi)
            else:               # When Y=0 is observed, H_i is surely true.
                N_Hi = expected_num_obs(1-p_Y1_Hi, target_p) #tmp/math.log2(1-p_Y1_Hi)
        else:
            N_Hi = (target_llr - priori_odds)/KL_Hi
        if math.isinf(KL_Hi_):  # An element of [p_Y1_Hi, 1 - p_Y1_Hi] is 0
            if p_Y1_Hi == 0:    # When Y=1 is observed, H_i_ is surely true.
                N_Hi_ = expected_num_obs(p_Y1_Hi_, target_p) #tmp/math.log2(p_Y1_Hi_)
            else:
                N_Hi_ = expected_num_obs(1 - p_Y1_Hi_, target_p) #tmp/math.log2(1-p_Y1_Hi_)
        else:
            N_Hi_ = (target_llr - priori_odds_)/KL_Hi_

    N_Hi_ave = p_Hi * N_Hi + p_Hi_ * N_Hi_

    if DEBUG:
        print(alpha)
        print(p_X0)
        print(p_X1_Hi)
        print(p_X1_Hi_)
        print(p_Y1_Hi)
        print(p_Y1_Hi_)

        print(f'Traget LLR: {target_llr}')
        print(f'{priori_odds=:f}, {priori_odds_=:f}')
        print(f'{KL_Hi=:f}, {KL_Hi_=:f}')

        print(N_Hi, N_Hi_, N_Hi_ave)
    return N_Hi_ave, N_Hi, N_Hi_

def main(hyp, hyp_, M_r, n_d):
    print(f'M_r, n_d, U, V, k_Hi, k_Hi_')
    list_min_N = []
    list_min_N_ = []
    for M_r in range(200, 300):
        for i in range(1, M_r.bit_length()):
            n_d = 2**i
            if 2*M_r % n_d == 0:
                list_N = []
                list_N_ = []
                for U in list_U10:
                    for V in list_V4:
                        N_Hi_ave, N_Hi, N_Hi_ = Hi_768(U, V, M_r, n_d, hyp, hyp_)
                        list_N.append(N_Hi)
                        list_N_.append(N_Hi_)
                        print(f'{M_r}, {n_d}, {U}, {V}, {N_Hi}, {N_Hi_}')
                print(f'{M_r}, {n_d}, {min(list_N)}, {min(list_N_)}')
                list_min_N.append(min(list_N))
                list_min_N_.append(min(list_N_))
    print(max(list_min_N))
    print(max(list_min_N_))

    # print(list_min_k)

def cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem):
    list_N_ave = []
    list_N = []
    list_N_ = []
    if mlkem == 768 or mlkem == 512:
        list_V = list_V4
        list_U = list_U10
    elif mlkem == 1024:
        list_V = list_V5
        list_U = list_U11

    for U in list_U:
        for V in list_V:
            N_Hi_ave, N_Hi, N_Hi_ = gen_N_Hi(U, V, M_r, n_d, hyp, hyp_, mlkem)
            list_N_ave.append((N_Hi_ave, U, V))
            list_N.append((N_Hi, U, V))
            list_N_.append((N_Hi_, U, V))
            #print(f'{U}, {V}, {N_Hi_ave}')
    min_N_ave = min(list_N_ave, key=lambda x: x[0])
    min_N = min(list_N, key=lambda x: x[0])
    min_N_ = min(list_N_, key=lambda x: x[0])
    # print(min_N_ave, min_N, min_N_)
    return min_N_ave[0]

def cal_N_obs_min_768():
    mlkem = 768
    for M_r in range(100,300):
        for i in range(1, M_r.bit_length()):
            n_d = 2**i
            if 2*M_r % n_d == 0:
                hyp, hyp_ = [-2,-1,0], [1,2]  ### H_0 null hypothesis and alternative hypothesis
                N_H0_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-2,-1], [0]  ### H_1 null hypothesis and alternative hypothesis
                N_H1_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [1], [2]  ### H_2 null hypothesis and alternative hypothesis
                N_H2_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-2], [-1]  ### H_3 null hypothesis and alternative hypothesis
                N_H3_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                # N_obs_min = (
                #     (1/16)*(N_H0_min + N_H1_min + N_H3_min) + 
                #     (4/16)*(N_H0_min + N_H1_min + N_H3_min_) +
                #     (6/16)*(N_H0_min + N_H1_min_) +
                #     (4/16)*(N_H0_min_ + N_H2_min) +
                #     (1/16)*(N_H0_min_ + N_H2_min_))

                N_obs_min = (N_H0_min + (11/16)*N_H1_min + (5/16)*N_H2_min + (5/16)*N_H3_min)

                print(f'{M_r}, {n_d}, {N_obs_min}')

def cal_N_obs_min_512():
    mlkem = 512
    for M_r in range(100,300):
        for i in range(1, M_r.bit_length()):
            n_d = 2**i
            if 2*M_r % n_d == 0:
                hyp, hyp_ = [-3,-2,-1,0], [1,2,3]  ### H_0 null hypothesis and alternative hypothesis
                N_H0_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-3,-2,-1], [0]  ### H_1 null hypothesis and alternative hypothesis
                N_H1_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [1], [2,3]  ### H_2 null hypothesis and alternative hypothesis
                N_H2_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-3,-2], [-1]  ### H_3 null hypothesis and alternative hypothesis
                N_H3_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [2], [3]  ### H_4 null hypothesis and alternative hypothesis
                N_H4_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-3], [-2]  ### H_5 null hypothesis and alternative hypothesis
                N_H5_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                N_obs_min = (N_H0_min + (42/64)*N_H1_min + (22/64)*N_H2_min + (22/64)*N_H3_min
                             + (7/64)*N_H4_min + (7/64)*N_H5_min)

                print(f'{M_r}, {n_d}, {N_obs_min}')

def cal_N_obs_min_1024():
    mlkem = 1024
    for M_r in range(100,300):
        for i in range(1, M_r.bit_length()):
            n_d = 2**i
            if 2*M_r % n_d == 0:
                hyp, hyp_ = [-2,-1,0], [1,2]  ### H_0 null hypothesis and alternative hypothesis
                N_H0_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-2,-1], [0]  ### H_1 null hypothesis and alternative hypothesis
                N_H1_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [1], [2]  ### H_2 null hypothesis and alternative hypothesis
                N_H2_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                hyp, hyp_ = [-2], [-1]  ### H_3 null hypothesis and alternative hypothesis
                N_H3_min = cal_N_Hi_min(M_r, n_d, hyp, hyp_, mlkem)

                N_obs_min = (N_H0_min + (11/16)*N_H1_min + (5/16)*N_H2_min + (5/16)*N_H3_min)

                print(f'{M_r}, {n_d}, {N_obs_min}')

if __name__ == '__main__':
    ### H_0
    hyp = [-2,-1,0]   ### null hypothesis
    hyp_ = [1,2]  ### alternative hypothesis
    ### H_1
    # hyp = [-2,-1]   ### null hypothesis
    # hyp_ = [0]  ### alternative hypothesis
    ### H_2
    # hyp = [1]   ### null hypothesis
    # hyp_ = [2]  ### alternative hypothesis
    ### H_3
    # hyp = [-2]   ### null hypothesis
    # hyp_ = [-1]  ### alternative hypothesis
    # U = 416
    # V = 416
    # M_r = 300
    # n_d = 2
    k, _, _= gen_N_Hi(211, 2497, 209, 2, hyp, hyp_, 768, True)

    # main(hyp, hyp_, M_r, n_d)
    #cal_N_obs_min_768()
