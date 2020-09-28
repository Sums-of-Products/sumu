#include <cstdint>

#define bm32 uint32_t
#define bm64 uint64_t

inline int dkbit_32(bm32 mask, int k) {
  //Drop kth bit
  if (mask == 0) {
    return mask;
  }
  bm32 trunc = mask >> (k + 1);
  trunc <<= k;
  return ((( (bm32) 1 << k) - 1) & mask) | trunc;
}

inline int ikbit_32(bm32 mask, int k, int bit) {
  // Shifts all bits >= k to the left and inserts bit to k
  bm32 newmask;
  if (k == 0) {
    newmask = mask << 1;
  }
  else {
    newmask = mask >> (k-1);
  }
  newmask ^= (-bit ^ newmask) & 1;
  newmask <<= k;
  return (newmask | ((1 << k) - 1)) & mask;
}

inline int lsb_32(bm32 x) {
  return ((x-1) & x) ^ x;
}
