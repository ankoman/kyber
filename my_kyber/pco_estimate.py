import math
from scipy.stats import norm
from decimal import *

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

if __name__ == '__main__':
    delta_w()
