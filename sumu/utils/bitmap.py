import numpy as np


def bm(ints, ix=None):
    if type(ints) not in [set, tuple]:
        ints = {int(ints)}
    ints = {int(x) for x in ints}
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**k
    return int(bitmap)


def bm_to_ints(bm):
    return tuple(i for i in range(len(format(bm, 'b')[::-1]))
                 if format(bm, 'b')[::-1][i] == "1")


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
