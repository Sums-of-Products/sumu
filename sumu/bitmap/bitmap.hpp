#include <cstdint>

#define bm32 uint32_t
#define bm64 uint64_t

inline bm32 dkbit_32(bm32 bitmap, int k) {
  //Drop kth bit
  if (bitmap == 0) {
    return bitmap;
  }
  bm32 trunc = bitmap >> (k + 1);
  trunc <<= k;
  return ((( (bm32) 1 << k) - 1) & bitmap) | trunc;
}

inline bm32 ikbit_32(bm32 bitmap, int k, int bit) {
  // Shifts all bits >= k to the left and inserts bit to k
  bm32 newmask;
  if (k == 0) {
    newmask = bitmap << 1;
  }
  else {
    newmask = bitmap >> (k-1);
  }
  newmask ^= (-bit ^ newmask) & 1;
  newmask <<= k;
  return newmask | (((1 << k) - 1) & bitmap);
}

inline int lsb_32(bm32 x) {
  return ((x-1) & x) ^ x;
}

inline int count_32(bm32 bitmap)
{
    int count = 0;
    while (bitmap) {
        count += bitmap & 1;
        bitmap >>= 1;
    }
    return count;
}


inline int fbit_32(bm32 mask)
// Get index of first 1-bit.
{
  int k{0};
  while ( (1 & mask) == 0) {
    k++;
    mask >>= 1;
  }
  return k;
}

inline bm32 kzon_32(bm32 bitmap, int k)
// Set kth 0-bit to 1.
{
  bm32 nbitmap = ~bitmap;
  for (int i = 0; i < k; ++i) {
    nbitmap &= ~(nbitmap & -nbitmap);
  }
  return bitmap | (nbitmap & -nbitmap);
}
