import math, random, copy
from tqdm import tqdm
from scipy.stats import norm, entropy
from decimal import *
import numpy as np
from test import test_ML_KEM, k, n, Rq, q

ZEROS = (0).to_bytes(32, 'big')

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

def hyp_test(inst, U, V, dk, mask_u, mask_v, i, j, M_r, n_d, hyp, hyp_, target_odds, mlkem, PK_MASK=False):
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))

    ### Uniform dist.
    list_index = np.array(list_i)*tau
    prob_U = 1 / len(list_index)
    if mlkem == 512:
        p_priori = np.array([20/16, 15/64, 6/64, 1/64, 1/64, 6/64, 15/64])   ### -1 = 15/64, -2 = 6/64, -3 = 1/64
        dv = 4
        sigma = math.sqrt(236*2)
    else:
        p_priori = np.array([6/16, 4/16, 1/16, 1/16, 4/16])   ### -1 = 4/16, -2 = 1/16
        if mlkem == 768:
            dv = 4
            sigma = math.sqrt(236*3)
        elif mlkem == 1024:
            dv = 5
            sigma = math.sqrt(236*4)
    mask = np.zeros_like(p_priori)
    mask[hyp + hyp_] = p_priori[hyp + hyp_]
    p_priori = mask / sum(mask)

    ### Probabilities
    p_X1_Hi = 0 ### Pr(X=1|H_i)
    p_X1_Hi_ = 0 ### Pr(X=1|H_i_)
    p_Hi = sum(p_priori[i] for i in hyp)
    p_Hi_ = sum(p_priori[i] for i in hyp_)

    if PK_MASK:
        for idx in list_index:
            for s in hyp:
                ### Exact case
                # for j in range(-3328//2**(dv+1), 3328//2**(dv+1)):
                #     p_X1_Hi += 1 - norm.cdf(832, idx - s*U + V + j, sigma)
                #     p_X1_Hi += norm.cdf(-832, idx - s*U + V + j, sigma)
                #p_X1_Hi *= 2**dv
                #p_X1_Hi /= (3328 + 2**dv)
                ### Approximate case
                p_X1_Hi += (1 - norm.cdf(832, idx - s*U + V, 65.5)) * (p_priori[s] / p_Hi)
                p_X1_Hi += norm.cdf(-832, idx - s*U + V, 65.5) * (p_priori[s] / p_Hi)
        p_X1_Hi *= prob_U 
        for idx in list_index:
            for s in hyp_:
                ### Exact case
                # for j in range(-3328//2**(dv+1), 3328//2**(dv+1)):
                #     p_X1_Hi_ += 1 - norm.cdf(832, idx - s*U + V + j, sigma)
                #     p_X1_Hi_ += norm.cdf(-832, idx - s*U + V + j, sigma)
                #p_X1_Hi_ *= 2**dv
                #p_X1_Hi_ /= (3328 + 2**dv)
                ### Approximate case
                p_X1_Hi_ += (1 - norm.cdf(832, idx - s*U + V, 65.5)) * (p_priori[s] / p_Hi_)
                p_X1_Hi_ += norm.cdf(-832, idx - s*U + V, 65.5) * (p_priori[s] / p_Hi_)
        p_X1_Hi_ *= prob_U 
    else:
        for idx in list_index:
            for s in hyp:
                if decode(idx - s*U + V):
                    p_X1_Hi += prob_U * p_priori[s] / p_Hi

        for idx in list_index:
            for s in hyp_:
                if decode(idx - s*U + V):
                    p_X1_Hi_ += prob_U * p_priori[s] / p_Hi_
    if p_X1_Hi > 1:
        p_X1_Hi = 1
    if p_X1_Hi_ > 1:
        p_X1_Hi_ = 1
    p_X0_Hi = 1 - p_X1_Hi ### Pr(X=0|H_i)
    p_X0_Hi_ = 1 - p_X1_Hi_ ### Pr(X=0|H_i_)

    beta = 0
    if mlkem == 512:
        e1_max = 3
    else:
        e1_max = 2
    for idx in list_index:
        if PK_MASK:
            ### Approximate case
            beta += (1 - norm.cdf(832, idx + e1_max*U, 65.5)) + norm.cdf(-832, idx + e1_max*U, 65.5)
            beta *= prob_U
        else:
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

    ### Make query
    u = copy.deepcopy(mask_u)
    v = copy.deepcopy(mask_v)
    u[j].coeff[i] = (mask_u[j].coeff[i] + U) % q
    c1 = Rq.polyvecCompEncode(u)
    v.coeff[0] = (mask_v.coeff[0] + V) % q
    while True:        
        n_query += 1
        ### query
        res = inst.dec_invalidRandCoef(dk, c1 + v.polyCompEncode(), M_r, n_d)

        ### Posteriori probability update
        if ZEROS != res:
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
    ### parameters
    PK_MASK = True
    M_r = 209
    n_d = 2
    target_odds = math.log2(0.99/0.01)
    print(f'Target odds: {target_odds}')

    # Prepare keys and public key mask
    #random.seed(0)
    row, pos, scalar, rot = 0, 0, 1, 0
    d = random.randint(0, 2**256-1).to_bytes(32, 'big')
    z = random.randint(0, 2**256-1).to_bytes(32, 'big')

    inst = test_ML_KEM()
    pk, sk = inst.cca_keygen(z, d)
    dk = sk[:384*k]
    s = np.array([Rq.intt(Rq.decode(dk[384*i:])).coeff for i in range(k)])

    # Generate pk mask
    mask_u, mask_v = [Rq() for i in range(k)], Rq()
    if PK_MASK:
        mask_u, mask_v = inst.get_pk_mask(sk, pos, row, scalar, rot)

    random.seed()
    list_recovered_s = [None] * n*k
    n_query = 0
    for j in tqdm(range(k)):
        for i in range(n):
            ### Null hypothesis: H0 vs. alternative hypothesis: H0_
            t, response = hyp_test(inst, 211, 2497, dk, mask_u, mask_v, i, j, M_r, n_d, [-2,-1,0], [1,2], target_odds, 768, PK_MASK)
            n_query += t
            if response:
                ### Null hypothesis: H1 vs. alternative hypothesis: H1_
                t, response = hyp_test(inst, 208, 416, dk, mask_u, mask_v, i, j, M_r, n_d, [-2,-1], [0], target_odds, 768, PK_MASK)
                n_query += t
                if response:
                    ### Null hypothesis: H3 vs. alternative hypothesis: H3_
                    t, response = hyp_test(inst, 107, 832, dk, mask_u, mask_v, i, j, M_r, n_d, [-2], [-1], target_odds, 768, PK_MASK)
                    n_query += t
                    if response:
                        recovered_s = -2
                    else:
                        recovered_s = -1
                else:
                    recovered_s = 0
            else:
                ### Null hypothesis: H2 vs. alternative hypothesis: H2_
                t, response = hyp_test(inst, 107, 2497, dk, mask_u, mask_v, i, j, M_r, n_d, [1], [2], target_odds, 768, PK_MASK)
                n_query += t
                if response:
                    recovered_s = 1
                else:
                    recovered_s = 2
            
            list_recovered_s[j*n + ((256-i) % 256)] = recovered_s if i == 0 else -recovered_s  ### Rajendra's method

    print(f'Required queries to achive the target odds for each coeffs: {n_query/(k*n)}')

    n_correct_coeff = 0
    for a, b in zip(list_recovered_s, s.flatten()):
        if a == b:
            n_correct_coeff += 1
    print(f'{n_correct_coeff=}')


if __name__ == '__main__':
    main()