import numpy as np


def bm(ints, idx=None):
    if idx is not None:
        ints = [idx.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**int(k)
    return bitmap


def bm_to_ints(bm):
    return tuple(i for i in range(len(format(bm, 'b')[::-1]))
                 if format(bm, 'b')[::-1][i] == "1")


def msb(n):
    blen = 0
    while (n > 0):
        n >>= 1
        blen += 1
    return blen


def bm_to_np64(bitmap, k=1):
    np64_seq = np.zeros(k, dtype=np.uint64)
    mask = (1 << 64) - 1
    for j in range(k):
        np64_seq[j] = (bitmap & mask) >> 64*j
        mask *= 2**64
    return np64_seq


def np64_to_bm(np64_seq):
    if type(np64_seq) in {np.uint64, int}:
        return int(np64_seq)
    pyint = 0
    for part in np64_seq[::-1]:
        pyint |= int(part)
        pyint *= 2**64
    pyint >>= 64
    return pyint
