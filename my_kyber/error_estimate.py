from my_ml_kem import *
import random, copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class test_ML_KEM(my_ML_KEM):
    def test_dec(self, dk: bytearray, c: bytearray, vv) -> bytearray:
        dk = dk[:384*k]
        u = Rq.polyvecDecodeDecomp(c)
        v = Rq.polyDecodeDecomp(c[32*du*k:])

        ntt_s = [Rq.decode(dk[384*i:]) for i in range(k)] 
        ntt_u = [Rq.ntt(u[i]) for i in range(k)]

        # Vector-vector multiplication
        stu = Rq()
        for i in range(k):
                stu = stu + (ntt_s[i] @ ntt_u[i])
        intt_stu = Rq.intt(stu)

        w = v - intt_stu

        return w, v

    def test_enc(self, pk: bytearray, m: bytearray) -> bytearray:
        Kr = hash_G(m + hash_H(pk))
        coin = Kr[32:]
        t_ = [Rq.decode(pk[12*32*i:]) for i in range(k)]
        rho = pk[-32:]
        N = 0

        At = self.genA(rho, True)

        y = []
        for i in range(k):
            y.append(Rq.sample_cbd(coin, N, eta1))
            N += 1

        e1 = []
        for i in range(k):
            e1.append(Rq.sample_cbd(coin, N, eta2))
            N += 1

        e2 = Rq.sample_cbd(coin, N, eta2)

        ntt_y = [Rq.ntt(y[i]) for i in range(k)]

        # Matrix-vector multiplication
        Aty = [Rq() for x in range(k)]
        for i in range(k):
            for j in range(k):
                Aty[i] = Aty[i] + (At[i][j] @ ntt_y[j])

        u = [Rq.intt(Aty[i]) + e1[i] for i in range(k)]
        # print(hex(self.unpack(e1)))

        mu = Rq.msgdecode(m)

        # Vector-vector multiplication
        tty = Rq()
        for i in range(k):
                tty = tty + (t_[i] @ ntt_y[i])
        v = Rq.intt(tty) + e2 + mu

        c1 = Rq.polyvecCompEncode(u)    ### 640 bytes

        c2 = v.polyCompEncode()         ### 128 bytes

        return c1 + c2, v

def show_histogram(data, label, gaussian = False):
    print(label)
    sigma = np.std(data, ddof=1)
    print(sigma)
    print(np.mean(data))

    ### Draw a histgram and a normal distribution
    count, bins, ignored = plt.hist(data, bins=100, density=True, alpha=0.6)
    if gaussian:
        x = np.linspace(min(bins), max(bins), 100)
        pdf = norm.pdf(x, 0, sigma)
        plt.plot(x, pdf, 'r-', lw=2, label='Normal dist')
        plt.title("Histogram with Normal Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # random.seed(0)
    list_E = []
    list_dv = []
    list_v = []
    for seed in range(1000):
        m = 0#random.randint(0, 2**256-1)
        d = random.randint(0, 2**256-1)
        z = random.randint(0, 2**256-1)

        tv_d = d.to_bytes(32, 'big')
        tv_z = z.to_bytes(32, 'big')
        tv_m = m.to_bytes(32, 'big')

        inst = test_ML_KEM()
        pk, sk = inst.cca_keygen(tv_z, tv_d)

        c, v = inst.test_enc(pk, tv_m)
        list_v.extend(v)

        w, vp = inst.test_dec(sk, c, v)
        list_E.extend(w)
        # print(v)
        # print(v*1)
        # print(vp)
        # print(np.array(v*1) - np.array(vp))
        list_dv.extend(np.array(v*1) - np.array(vp))
    
    show_histogram(list_E, 'E', True)

    show_histogram(list_dv, 'dv')

    show_histogram(list_v, 'v')

if __name__ == '__main__':
    main()