import numpy as np
import gadget


def bm(ints, ix=None):
    if type(ints) not in [set, tuple]:
        ints = {int(ints)}
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**k
    return int(bitmap)  # without the cast np.int64 might sneak in somehow and break drv


def subsets_size_k(k, n):
    if k == 0:
        yield 0
        return
    #S = (1 << k) - 1
    S = 2**k - 1
    # limit = (1 << n)
    limit = 2**n
    while S < limit:
        yield S
        x = S & -S
        r = S + x
        S = (((r ^ S) >> 2) // x) | r


def ikbit(mask, k, bit):
    """shift all bits >=k to the left and insert bit to k"""
    if k == 0:
        # newmask = mask << 1
        newmask = mask * 2
    else:
        newmask = mask >> k-1
    newmask ^= (-bit ^ newmask) & 1
    #newmask <<= k
    newmask *= 2**k
    #return newmask | ((1 << k) - 1) & mask
    return newmask | (2**k - 1) & mask


psets = list()
for pset in subsets_size_k(2, 5):
    C = [0, 2, 4, 6, 7]
    for v in C:
        pset = ikbit(pset, v, 0)
    psets.append(pset)

psets = np.array(psets, dtype=np.uint64)
weights = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], dtype=np.float64)

U = 872
T = 320
U_bin = bin(U)[2:]
T_bin = bin(T)[2:]
print("U\n{}".format('0'*(10-len(U_bin)) + U_bin))
print("T\n{}".format('0'*(10-len(T_bin)) + T_bin))
print("Pset\t\tW\tU\tT")
for i in range(len(psets)):
    pset = bin(psets[i])[2:]
    print("{}\t{}\t{}\t{}".format('0'*(10-len(pset)) + pset,
                                  weights[i],
                                  int(psets[i]) & U == psets[i],
                                  int(psets[i]) & T > 0))

print(gadget.weight_sum_contribs(-float("inf"), psets, weights, 10, U, T, 10))
print(gadget.weight_sum(-float("inf"), psets, weights, 10, U, T, 10))
