import math, random
from tqdm import tqdm
from scipy.stats import norm, entropy
from decimal import *
import numpy as np
from my_ml_kem import *

def decode(val):
    val = val % 3329
    if val >= 832 and val < 2496:
        return True
    else:
        return False

def get_without_i(lst, idx):
    copied = lst.copy()
    copied.pop(idx)
    return copied

def hyp_test(U, V, target_s, non_target_coeffs, M_r, n_d, hyp, hyp_, target_odds, mlkem):
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

    ### Probabilities
    p_X1_Hi = 0 ### Pr(X=1|H_i)
    p_Hi = sum(p_priori[i] for i in hyp)
    for idx in list_index:
        for s in hyp:
            if decode(idx - s*U + V):
                p_X1_Hi += prob_U * p_priori[s] / p_Hi
    if p_X1_Hi > 1:
        p_X1_Hi = 1

    p_X1_Hi_ = 0 ### Pr(X=1|H_i_)
    p_Hi_ = sum(p_priori[i] for i in hyp_)
    for idx in list_index:
        for s in hyp_:
            if decode(idx - s*U + V):
                p_X1_Hi_ += prob_U * p_priori[s] / p_Hi_
    if p_X1_Hi_ > 1:
        p_X1_Hi_ = 1

    beta = 0
    if mlkem == 512:
        e1_max = 3
    else:
        e1_max = 2
    for idx in list_index:
        if decode(idx + e1_max*U):
            beta += prob_U
    alpha = 1 - (1 - beta)**32 ###  Pr(Y=1)

    p_Y1_Hi = alpha + (1-alpha)*p_X1_Hi ### Pr(Y=1|H_i)
    p_Y1_Hi_ = alpha + (1-alpha)*p_X1_Hi_ ### Pr(Y=1|H_i_)
    p_Y0_Hi = 1 - p_Y1_Hi
    p_Y0_Hi_ = 1 - p_Y1_Hi_

    p_Hi_Y = p_Hi
    p_Hi__Y = p_Hi_
    n_query = 0
    while True:        
        n_query += 1
        ### Target coefficient effect
        r = random.randint(-n_d//2, n_d//2)
        DF_response = decode(V - target_s * U + r*tau)

        ### Non-target coefficients effect
        non_target_m = [decode(-elem * U + random.randint(-n_d//2, n_d//2)*tau) for elem in non_target_coeffs]
        if 1 in non_target_m:
            DF_response = True

        ### Posteriori probabiliry update
        if DF_response:
            p_Hi_Y = (p_Y1_Hi * p_Hi_Y) / (p_Y1_Hi * p_Hi_Y + p_Y1_Hi_ * p_Hi__Y)   
            p_Hi__Y = (p_Y1_Hi_ * p_Hi__Y) / (p_Y1_Hi * p_Hi_Y + p_Y1_Hi_ * p_Hi__Y)
        else:
            p_Hi_Y = (p_Y0_Hi * p_Hi_Y) / (p_Y0_Hi * p_Hi_Y + p_Y0_Hi_ * p_Hi__Y)   
            p_Hi__Y = (p_Y0_Hi_ * p_Hi__Y) / (p_Y0_Hi * p_Hi_Y + p_Y0_Hi_ * p_Hi__Y)

        ### Likelihood check
        if p_Hi__Y == 0 or (p_Hi_Y != 0 and math.log2(p_Hi_Y/p_Hi__Y) >= target_odds):
            return n_query, True
        if p_Hi_Y == 0 or math.log2(p_Hi__Y/p_Hi_Y) >= target_odds:
            return n_query, False



def main():
    #random.seed(0)
    d = random.randint(0, 2**256-1).to_bytes(32, 'big')
    inst = my_ML_KEM()
    pk, sk = inst.cpa_keygen(d)
    s = np.array([Rq.intt(Rq.decode(sk[384*i:])).coeff for i in range(k)])

    M_r = 209
    n_d = 2
    target_odds = math.log2(0.99/0.01)
    print(f'Target odds: {target_odds}')

    list_recovered_s = []
    n_query = 0
    for j in tqdm(range(k)):
        for i in range(n):
            target_s = s[j][i]
            non_target_coeffs = get_without_i(s[j].tolist(), i)
            ### Null hypothesis: H0 vs. alternative hypothesis: H0_
            t, response = hyp_test(211, 2497, target_s, non_target_coeffs, M_r, n_d, [-2,-1,0], [1,2], target_odds, 768)
            n_query += t
            if response:
                ### Null hypothesis: H1 vs. alternative hypothesis: H1_
                t, response = hyp_test(208, 416, target_s, non_target_coeffs, M_r, n_d, [-2,-1], [0], target_odds, 768)
                n_query += t
                if response:
                    ### Null hypothesis: H3 vs. alternative hypothesis: H3_
                    t, response = hyp_test(107, 832, target_s, non_target_coeffs, M_r, n_d, [-2], [-1], target_odds, 768)
                    n_query += t
                    if response:
                        list_recovered_s.append(-2)
                    else:
                        list_recovered_s.append(-1)
                else:
                    list_recovered_s.append(0)
            else:
                ### Null hypothesis: H2 vs. alternative hypothesis: H2_
                t, response = hyp_test(107, 2497, target_s, non_target_coeffs, M_r, n_d, [1], [2], target_odds, 768)
                n_query += t
                if response:
                    list_recovered_s.append(1)
                else:
                    list_recovered_s.append(2)

    print(f'Required queries to achive the target odds for each coeffs: {n_query/(k*n)}')

    n_correct_coeff = 0
    for a, b in zip(list_recovered_s, s.flatten()):
        if a == b:
            n_correct_coeff += 1
    print(f'{n_correct_coeff=}')


if __name__ == '__main__':
    main()