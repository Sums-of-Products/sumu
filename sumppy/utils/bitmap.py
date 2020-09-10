import numpy as np


def bm(ints, ix=None):
    if type(ints) not in [set, tuple]:
        ints = {int(ints)}
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**k
    return int(bitmap)  # without the cast np.int64 might sneak in somehow and break drv


def bm_to_ints(bm):
    return tuple(i for i in range(len(format(bm, 'b')[::-1]))
                 if format(bm, 'b')[::-1][i] == "1")


def translate_psets_to_bitmaps(C, scores):
    K = len(C[0])
    scores_list = list()
    for v in sorted(scores):
        tmp = [-float('inf')]*2**K
        for pset in scores[v]:
            tmp[bm(set(pset), ix=C[v])] = scores[v][pset]
        scores_list.append(tmp)
    return scores_list


def msb(n):
    blen = 0
    while (n > 0):
        n >>= 1
        blen += 1
    return blen


def bm_to_pyint_chunks(bitmap, minwidth=1):
    chunk = [0]*max(minwidth, (msb(bitmap)-1)//64+1)
    if len(chunk) == 1:
        return bitmap
    mask = (1 << 64) - 1
    for j in range(len(chunk)):
        chunk[j] = (bitmap & mask) >> 64*j
        mask *= 2**64
    return chunk


def bm_to_np64(bitmap):
    chunk = np.zeros(max(1, (msb(bitmap)-1)//64+1), dtype=np.uint64)
    mask = (1 << 64) - 1
    for j in range(len(chunk)):
        chunk[j] = (bitmap & mask) >> 64*j
        mask *= 2**64
    return chunk


def bms_to_np64(bitmaps, minwidth=1):
    blen = np.array([msb(x) for x in bitmaps])
    dim1 = len(bitmaps)
    dim2 = max(minwidth, max((blen - 1) // 64) + 1)
    if dim2 == 1:
        return np.array(bitmaps, dtype=np.uint64)
    chunks = np.zeros(shape=(dim1, dim2), dtype=np.uint64)
    for i in range(dim1):
        n_c = (blen[i] - 1)//64
        mask = (1 << 64) - 1
        for j in range(n_c + 1):
            chunks[i][j] = (bitmaps[i] & mask) >> 64*j
            mask *= 2**64
    return chunks


def np64_to_bm(chunk):
    if type(chunk) in {np.uint64, int}:
        return int(chunk)
    bm = 0
    for part in chunk[::-1]:
        bm |= int(part)
        bm *= 2**64
    bm >>= 64
    return bm


def fbit(mask):
    """get index of first set bit"""
    k = 0
    while 1 & mask == 0:
        k += 1
        mask >>= 1
    return k


def kzon(mask, k):
    """set kth zerobit on"""
    nmask = ~mask
    for i in range(k):
        nmask &= ~(nmask & -nmask)
    return mask | (nmask & -nmask)


def dkbit(mask, k):
    """drop kth bith"""
    if mask == 0:
        return mask
    trunc = mask >> (k+1)
    #trunc <<= k
    trunc *= 2**k
    # return ((1 << k) - 1) & mask | trunc
    return (2**k - 1) & mask | trunc


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


def ssets(mask):
    S = mask
    while S > 0:
        yield S
        S = (S - 1) & mask
