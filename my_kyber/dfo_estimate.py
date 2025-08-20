import math
from scipy.stats import norm
from decimal import *

def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def clip(x, th):
    if x < th:
        return x
    else:
        return th

def h(p,q):
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)

    return -p*math.log2(p)-q*math.log2(q)

def false_positive():
    n=256
    k=3
    getcontext().prec = 150
    prob = math.log2(Decimal(1)-(Decimal(1)-(Decimal(1)/(Decimal(1024)**Decimal(14))))**Decimal(4*n*k))
    print(prob)

if __name__ == '__main__':
    false_positive()

    sigma = 51
    M_r = 256
    n_d = 32
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))
    d = max(list_i)
    list_G = []

    for b in range(104):
        omega = 832 + b

        pX0 = norm.cdf(832, omega, sigma)
        pX1 = 1 - pX0
        
        pY0 = 0
        p11 = 0
        for i in list_i:
            pY0 += norm.cdf(832, i*tau+omega, sigma)
            p11 += clip(1 - norm.cdf(832, i*tau + omega, sigma), pX1)

        pY0 /= (n_d + 1)
        pY1 = 1 - pY0
        p11 /= (n_d + 1)
        pY1X0 = (pY1 - p11) / pX0
        pY1X1 = p11/pX1

        H_YX = pX0*h2(pY1X0) + pX1*h2(pY1X1)
        H_Y = h2(pY0)
        C = H_Y - H_YX
        G = 1/C
        list_G.append(G)

        print(f'{b}: {H_YX=}, {H_Y=}, {C=}, {G=}')
    print(f'min G: {min(list_G)}')
