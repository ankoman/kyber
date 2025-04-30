from my_ml_kem import *
import random

class test_ML_KEM(my_ML_KEM_512):
    def cca_dec_out_mp(self, c: bytearray, sk: bytearray):
        dk = sk[:384*k]
        pk = sk[384*k:768*k+32]
        h = sk[768*k+32:768*k+64]
        z = sk[768*k+64:768*k+96]
        mp = self.dec(dk, c)

        Kprp = hash_G(mp + h)
        Kp = Kprp[:32]
        rp = Kprp[32:]
        K_bar = hash_J(z + c)
        cp = self.enc(pk, mp, rp)
        
        if c != cp:
            Kp = K_bar

        return Kp, mp
    
    def pk_masked_cca_dec(self, c: bytearray, sk: bytearray, row: int, scalar: int, offset, rot: int):
        ### Decode
        pk = sk[384*k:768*k+32]
        A_ = self.genA(pk[-32:])
        A_star = [Rq.intt(A_[row][i]) * scalar for i in range(k)]
        A_star[0].coeff = np.roll(np.array(A_star[0].coeff), rot)
        A_star[0].coeff[0] *= -1
        A_star[1].coeff = np.roll(np.array(A_star[1].coeff), rot)
        A_star[1].coeff[0] *= -1

        t_ = [Rq.decode(pk[12*32*i:]) for i in range(k)]
        t_star = Rq.intt(t_[row]) * scalar
        t_star.coeff = np.roll(np.array(t_star.coeff), rot)
        #t_star.coeff[0] *= -1

        u = np.array(Rq.polyvecDecodeDecomp(c))
        v = Rq.polyDecodeDecomp(c[32*du*k:])

        u += A_star
        v += t_star

        ### Encode
        c1 = Rq.polyvecCompEncode(u)    ### 640 bytes
        c2 = v.polyCompEncode()         ### 128 bytes

        return self.cca_dec_out_mp(c1+c2, sk)


for seed in range(100):
    random.seed(seed)
    m = random.randint(0, 2**256-1)
    d = random.randint(0, 2**256-1)
    z = random.randint(0, 2**256-1)
    row = 1

    tv_d = d.to_bytes(32)
    tv_z = z.to_bytes(32)
    tv_m = m.to_bytes(32)

    inst = test_ML_KEM()
    pk, sk = inst.cca_keygen(tv_z, tv_d)

    c, K = inst.cca_enc(pk, tv_m)
    # print(f'  Bob (enc) side shared secret K: {K}')

    K, mp = inst.pk_masked_cca_dec(c, sk, row, 1, 0, 1)
    # print(f'Alice (dec) side shared secret K: {K}')

    # print(m)
    # print(int.from_bytes(mp, 'big'))
    print(seed, int.bit_count(m ^ int.from_bytes(mp, 'big')))
