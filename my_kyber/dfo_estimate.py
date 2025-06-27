import math
from scipy.stats import norm
from decimal import *

def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def clip(x):
    if x < 0.5:
        return x
    else:
        return 0.5

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


    p11 = 0
    p00 = 0
    p0 = 0

    b_range = 0
    sigma_b = 51
    M_r = 256
    n_d = 256
    tau = 2*M_r//n_d
    list_i = list(range(-n_d//2, n_d//2+1))
    d = max(list_i)
    c = -d*tau

    for i in list_i:
        p11 += clip(1 - norm.cdf(-c, i*tau, sigma_b))
        p00 += clip(norm.cdf(-c, i*tau, sigma_b))
        p0 += norm.cdf(-c, i*tau, sigma_b)
    
    p11 /= (n_d + 1)
    p00 /= (n_d + 1)
    p0 /= (n_d + 1)
    p1 = 1 - p0
    p10 = p1 - p11
    p01 = p0 - p00

    print(f'{p11=}, {p10=}, {p01=}, {p00=}, {p0=}, {p1=}')

    H_XY = p0*h(p00,p01) + p1*h(p10,p11)
    H_X = 1
    C = H_X - H_XY
    print(f'{H_XY=}, {H_X=}, {C=}')
