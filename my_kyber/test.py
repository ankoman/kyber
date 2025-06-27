from my_ml_kem import *
import numpy as np
import random, copy, math

class test_ML_KEM(my_ML_KEM):
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
    
    def get_pk_mask(self, sk: bytearray, pos: int, row: int, scalar: int, rot: int):
        ### Decode
        pk = sk[384*k:768*k+32]
        A_ = self.genA(pk[-32:])
        delta_u = [Rq.intt(A_[row][i]) * scalar for i in range(k)]
        for i in range(k):
            xtimes(delta_u[i], rot)

        t_ = [Rq.decode(pk[12*32*i:]) for i in range(k)]
        delta_v = Rq.intt(t_[row]) * scalar
        xtimes(delta_v, rot)

        return delta_u, delta_v ### Not compressed
    
    def pk_masked_cca_dec(self, c: bytearray, sk: bytearray, pos: int, row: int, scalar: int, rot: int):
        ### Decode
        pk = sk[384*k:768*k+32]
        A_ = self.genA(pk[-32:])
        A_star = [Rq.intt(A_[row][i]) * scalar for i in range(k)]
        xtimes(A_star[0], rot)
        xtimes(A_star[1], rot)
        xtimes(A_star[2], rot)

        t_ = [Rq.decode(pk[12*32*i:]) for i in range(k)]
        t_star = Rq.intt(t_[row]) * scalar
        xtimes(t_star, rot)

        ### pk mask
        u = np.array(Rq.polyvecDecodeDecomp(c))
        v = Rq.polyDecodeDecomp(c[32*du*k:])
        # v.coeff[0] = v.coeff[0] + 832
        # u[0].coeff[0] = u[0].coeff[0] + 100

        # A_star = Rq.polyvecCompEncode(A_star) 
        # t_star = t_star.polyCompEncode() 
        # c = A_star + t_star
        # A_star = Rq.polyvecDecodeDecomp(c)
        # A_star[0] *= scalar
        # A_star[1] *= scalar
        # xtimes(A_star[0], rot)
        # xtimes(A_star[1], rot)
        # t_star = Rq.polyDecodeDecomp(c[32*du*k:]) * scalar
        # xtimes(t_star, rot)
        u = A_star
        v = t_star

        ### Encode
        c1 = Rq.polyvecCompEncode(u)    ### 640 bytes
        c2 = v.polyCompEncode()         ### 128 bytes

        check = Rq.polyvecDecodeDecomp(c1+c2)
        pk_mask_check(check[pos], A_, pos, row)

        return self.cca_dec_out_mp(c1+c2, sk)

def comp(val: int):
    ### 10 bit comp
    if val < 0:
        val += q
    val <<= 10
    val += 1665
    val *= 1290167
    val >>= 32
    return val & 0x3ff

def decomp(val: int):
    ### 10 bit decomp
    if val < 0:
        val += q
    val &= 0x3ff
    val *= q
    val += 512
    val >>= 10
    return val

def approx(val: int):
    return decomp(comp(val))

def xtimes(poly: Rq, p: int):
    for i in range(p):
        coeff = poly.coeff.pop(0)
        coeff = coeff * -1
        poly.coeff.append(coeff if coeff > 0 else coeff + q)

def xtimes_approx(poly: Rq, p: int):
    for i in range(p):
        coeff = poly.coeff.pop(0)
        coeff = approx(coeff * -1)
        poly.coeff.append(coeff if coeff > 0 else coeff + q)

def approx_check():
    for i in range(3329):
        a = approx(q - i)
        b = q - approx(i)
        print(a-b)



def diffCount(list_a, list_b):
    cnt = 0
    for i in range(256):
        if list_a[i] != list_b[i]:
            cnt += 1
    return cnt

def pco_512_tanaka(inst, dk, d_u, d_v, i, j):
    V = 208
    U = 276
    zero = (0).to_bytes(32, 'big')

    # Make query U
    u = copy.deepcopy(d_u)
    v = copy.deepcopy(d_v)
    tmp_v = v.coeff[j]
    u[i].coeff[0] = (u[i].coeff[0] + U) % q
    c1 = Rq.polyvecCompEncode(u)

    # Make query V
    v.coeff[j] = (tmp_v + 3*V) % q
    if zero != inst.dec(dk, c1 + v.polyCompEncode()):
        v.coeff[j] = (tmp_v + 2*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            v.coeff[j] = (tmp_v +  V) % q
            if zero != inst.dec(dk, c1 + v.polyCompEncode()):
                return -3
            else: 
                return -2
        else:
            return -1
    else:
        v.coeff[j] = (tmp_v + -2*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            v.coeff[j] = (tmp_v + -1*V) % q
            if zero != inst.dec(dk, c1 + v.polyCompEncode()):
                return 3
            else:
                return 2
        else:
            v.coeff[j] = (tmp_v + -3*V) % q
            if zero != inst.dec(dk, c1 + v.polyCompEncode()):
                return 1
            else:
                return 0

def pco_768_tanaka(inst, dk, d_u, d_v, i, j):
    V = 208
    U = 276
    zero = (0).to_bytes(32, 'big')

    # Make query U
    u = copy.deepcopy(d_u)
    v = copy.deepcopy(d_v)
    tmp_v = v.coeff[j]
    u[i].coeff[0] = (u[i].coeff[0] + U) % q
    c1 = Rq.polyvecCompEncode(u)

    # Make query V
    v.coeff[j] = (tmp_v + 3*V) % q
    if zero != inst.dec(dk, c1 + v.polyCompEncode()):
        v.coeff[j] = (tmp_v + 2*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            return -2
        else:
            return -1
    else:
        v.coeff[j] = (tmp_v + -3*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            v.coeff[j] = (tmp_v + -2*V) % q
            if zero != inst.dec(dk, c1 + v.polyCompEncode()):
                return 2
            else:
                return 1
        else:
            return 0

def pco_768_rajendran(inst, dk, d_u, d_v, i, j):
    V = 208
    U = 208
    zero = (0).to_bytes(32, 'big')

    # Make query U
    u = copy.deepcopy(d_u)
    v = copy.deepcopy(d_v)
    tmp_v = v.coeff[0]
    u[i].coeff[j] = (u[i].coeff[j] + U) % q
    c1 = Rq.polyvecCompEncode(u)

    # Make query V
    v.coeff[0] = (tmp_v + 4*V) % q
    if zero != inst.dec(dk, c1 + v.polyCompEncode()):
        v.coeff[0] = (tmp_v + 3*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            return -2
        else:
            return -1
    else:
        v.coeff[0] = (tmp_v + 6*V) % q
        if zero != inst.dec(dk, c1 + v.polyCompEncode()):
            v.coeff[0] = (tmp_v + 5*V) % q
            if zero != inst.dec(dk, c1 + v.polyCompEncode()):
                return 0
            else:
                return 1
        else:
            return 2

def pco():
    # random.seed(1)
    PK_MASK = True
    row = 0
    pos = 0
    scalar = 100
    rot = 0
    # rot = random.randint(0, 511)
    # scalar = random.randint(1, 415)
    d = random.randint(0, 2**256-1)
    z = random.randint(0, 2**256-1)

    tv_d = d.to_bytes(32, 'big')
    tv_z = z.to_bytes(32, 'big')

    # Prepare secret key
    inst = test_ML_KEM()
    pk, sk = inst.cca_keygen(tv_z, tv_d)
    dk = sk[:384*k]
    s = [Rq.intt(Rq.decode(dk[384*i:])) for i in range(k)]

    # Generate pk mask
    d_u, d_v = [Rq() for i in range(k)], Rq()
    if PK_MASK:
        d_u, d_v = inst.get_pk_mask(sk, pos, row, scalar, rot)
        # d_u[0].coeff[0] += 10

    for i in range(1):
        attacked_key = [None] * 256
        for j in range(256):
            key = pco_768_rajendran(inst, dk, d_u, d_v, i, j)
            # attacked_key[j] = key              ### Tanaka's method
            attacked_key[(256-j) % 256] = key if j == 0 else -key  ### Rajendra's method
        
        if s[i].coeff == attacked_key:
            print(f"Attack success s[{i}]")
        else:
            print(f"Failed s[{i}]: {diffCount(s[i].coeff, attacked_key)}")
        # print(attacked_key)
        # print(s[i])
        
def pk_mask_check(ct_poly: Rq, A_, pos: int, row: int):
    flag = 0
    pk_poly = Rq.intt(A_[row][pos]) * 1 ### Make values positive

    a0_inv = 0
    for base in range(512):
        if pk_poly.coeff[base] != 0:
            a0_inv = pow(pk_poly.coeff[base], q-2, q)
            break

    for rot in range(256):
        xtimes_approx(ct_poly, 1)

        center = ct_poly.coeff[base] % q
        list_candidate = [center]
        offsets = [-2, 2, -1, 1]
        for offset in offsets:
            candidate = (center + offset) % q
            if approx(candidate) == center:
                list_candidate.append(candidate)

        for candidate in list_candidate:
            cnt_invalid = 0
            c = a0_inv * candidate % q
            if c < 416 or c > 2913: ### Negative values must be considered
                for i in range(256):
                    v = comp(c * pk_poly.coeff[(i+base) % 256] % q)
                    if v == comp(ct_poly.coeff[(i+base) % 256]): ### We can use compressed ciphertext
                        cnt_invalid += 1
                if cnt_invalid > 10:
                    print(f'{cnt_invalid=}, {rot=}, invalid ', end='')
                    if rot > 256:
                        print("invalid rot")
                    flag = 1
                    break
        else:
            continue
        break
    if flag == 0:
        print("undetected")
        input()
        
def main():
    for seed in range(1000):
        #random.seed(0)
        rot = random.randint(0, 511)
        scalar = random.randint(0, 416)
        m = 0#random.randint(0, 2**256-1)
        d = random.randint(0, 2**256-1)
        z = random.randint(0, 2**256-1)
        row = 0
        pos = 0

        tv_d = d.to_bytes(32, 'big')
        tv_z = z.to_bytes(32, 'big')
        tv_m = m.to_bytes(32, 'big')

        inst = test_ML_KEM()
        pk, sk = inst.cca_keygen(tv_z, tv_d)

        c, K = inst.cca_enc(pk, tv_m)
        # print(f'  Bob (enc) side shared secret K: {K}')

        K, mp = inst.pk_masked_cca_dec(c, sk, pos, row, scalar, rot)
        # print(f'Alice (dec) side shared secret K: {K}')

        # print(m)
        # print(int.from_bytes(mp, 'big'))
        wrong_bits = int.bit_count(m ^ int.from_bytes(mp, 'big'))
        hw_mp = int.bit_count(int.from_bytes(mp, 'big'))
        if hw_mp != 300:
            print(seed, hex(m), hex(d), rot, scalar, wrong_bits, hw_mp)


if __name__ == '__main__':
    random.seed(0)
    # main()
    for i in range(10):
        pco()