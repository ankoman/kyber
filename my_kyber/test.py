from my_ml_kem import *
import random, copy

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
    
    def pk_masked_cca_dec(self, c: bytearray, sk: bytearray, pos: int, row: int, scalar: int, rot: int):
        ### Decode
        pk = sk[384*k:768*k+32]
        A_ = self.genA(pk[-32:])
        A_star = [Rq.intt(A_[row][i]) * scalar for i in range(k)]
        # xtimes(A_star[0], rot)
        # xtimes(A_star[1], rot)

        t_ = [Rq.decode(pk[12*32*i:]) for i in range(k)]
        t_star = Rq.intt(t_[row]) * scalar
        # xtimes(t_star, rot)

        ### pk mask
        u = np.array(Rq.polyvecDecodeDecomp(c))
        v = Rq.polyDecodeDecomp(c[32*du*k:])
        A_star = Rq.polyvecCompEncode(A_star) 
        t_star = t_star.polyCompEncode() 
        c = A_star + t_star
        A_star = Rq.polyvecDecodeDecomp(c)
        A_star[0] *= scalar
        A_star[1] *= scalar
        xtimes(A_star[0], rot)
        xtimes(A_star[1], rot)
        t_star = Rq.polyDecodeDecomp(c[32*du*k:]) * scalar
        xtimes(t_star, rot)
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

def pk_mask_check(ct_poly: Rq, A_, pos: int, row: int):
    flag = 0
    pk_poly = Rq.intt(A_[row][pos]) * 1 ### Make values positive

    a0_inv = 0
    for base in range(512):
        if pk_poly.coeff[base] != 0:
            a0_inv = pow(pk_poly.coeff[base], q-2, q)
            break

    for rot in range(512):
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
            if c < 416: ### Negative values must be considered
                for i in range(256):
                    v = comp(c * pk_poly.coeff[(i+base) % 256] % q)
                    if v == comp(ct_poly.coeff[(i+base) % 256]): ### We can use compressed ciphertext
                        cnt_invalid += 1
                if cnt_invalid > 10:
                    print(f'{cnt_invalid=}, {rot=}, invalid ', end='')
                    flag = 1
                    break
        else:
            continue
        break
    if flag == 0:
        print("undetected")

def approx_check():
    for i in range(3329):
        a = approx(q - i)
        b = q - approx(i)
        print(a-b)

def prob():
    from decimal import *
    getcontext().prec = 150
    math.log2(Decimal(1)-(Decimal(1)-(Decimal(1)/(Decimal(q)**Decimal(13))))**Decimal(2*q*n*k))

for seed in range(10000):
    #random.seed(0)
    rot = random.randint(0, 511)
    scalar = random.randint(1, 415)
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

    K, mp = inst.pk_masked_cca_dec(c, sk, pos, row, 1, seed)
    # print(f'Alice (dec) side shared secret K: {K}')

    # print(m)
    # print(int.from_bytes(mp, 'big'))
    wrong_bits = int.bit_count(m ^ int.from_bytes(mp, 'big'))
    hw_mp = int.bit_count(int.from_bytes(mp, 'big'))
    print(seed, hex(m), hex(d), rot, scalar, wrong_bits, hw_mp)
