n = 2
k = 2
q = 3329
eta1 = 3
eta2 = 2
du = 10
dv = 4

import random, math

class Rq:
    """
    Name:        Rq
    12 bits * 256 = 384 bytes
    """

    def __init__(self):
        self.coeff = [0 for x in range(n)]

    def __repr__(self):
        return str(list(map(hex, self.coeff)))

    def __getitem__(self, index):
        return self.coeff[index]

    def __eq__(self, other):
        return self.coeff == other.coeff

    def __add__(self, other):
        tmp = self.__class__()
        for i in range(n):
            tmp.coeff[i] = (self.coeff[i] + other.coeff[i]) % q ### reduction
            #上の%qで値が正になるので、ref実装に合わせるため[-q/2,q/2]の範囲に戻す。
            # if tmp.coeff[i] > 1664:
            #     tmp.coeff[i] -= q 
        return tmp

    def __sub__(self, other):
        tmp = self.__class__()
        for i in range(n):
            tmp.coeff[i] = (self.coeff[i] - other.coeff[i]) % q ### reduction
            #上の%qで値が正になるので、ref実装に合わせるため[-q/2,q/2]の範囲に戻す。
            # if tmp.coeff[i] > 1664:
            #     tmp.coeff[i] -= q 
        return tmp

    @staticmethod
    def gen_random():
        x = Rq()
        for i in range(n):
            x.coeff[i] = random.randint(0, q-1)
        return x

def A2B(x0, x1):
    x = x0 + x1
    z0 = random.randint(0, 2**12-1)
    z1 = x ^ z0
    return z0, z1

### Must use round function
def ModSwitch(x0, x1):
    y0 = (x0 * 2**11) // q
    y1 = (x1 * 2**11) // q
    return y0, y1

# x0 = random.randint(0, q-1)
# x1 = random.randint(0, q-1)
# x = (x0 + x1) % q
# print(f'{x0=} + {x1=} = {x=}')  # A maskによる係数
# y0, y1 = A2B(x0, x1)
# print(f'{y0=} ^ {y1=} = {(y0^y1)%q=}')  # B maskによる係数

# ### Comp1
# y0, y1 = ModSwitch(x0, x1)
# print(f'{y0=} + {y1=} = {(y0+y1)%q=}')  # A maskによる係数
# y0, y1 = A2B(y0, y1)

def sanityCheck(val, tau):
    r = random.randint(0, q-1)
    t1 = (val - (832-tau)) % q
    t2 = 2*t1 % q
    if t2 < 4*tau:
        print('invalid')
    else:
        print('valid')

def ShowHWProb():
    sum = 0
    for i in range(129):
        c = math.comb(256, i)
        p = c / (2**256)
        sum += p
        print(f'{i=}, {p=}, {sum=}, 2^{math.log2(sum):.5}')
    
tau = 100
r = 2596
print(f'{r=}, {832-tau <= r and r < 832+tau}, {2496-tau < r and r < 2496+tau}')
sanityCheck(r, 100)

# ShowHWProb()

p = 0.5
N = 5500
for a in range(1,20):
    correctInEq_rate = 1 - p + (p/(2**a))
    incorrectInEq_rate = p/(2**a)
    availableInEq_rate = 1 - p + (p/(2**(a-1)))
    correct_rate = correctInEq_rate / availableInEq_rate
    sum = 0
    for i in range(2,a+1):
        sum += i/2**(i-1)
    queryFor1ineq = a*(1-p) + p*sum + p*a/2**(a-1)

    print(f'{queryFor1ineq=}')
    print(f'{a=}, {correct_rate=}')
