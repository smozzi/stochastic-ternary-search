// Partial Fisher-Yates: exact uniform ordered subset, O(B) rather than O(N).
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>
struct BRng {
  uint64_t s[4];
  static uint64_t rot(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  explicit BRng(uint64_t seed) {
    for (auto &a : s) {
      uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      a = z ^ (z >> 31);
    }
  }
  uint64_t next() {
    uint64_t r = rot(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rot(s[3], 45);
    return r;
  }
  uint64_t bounded(uint64_t n) {
    uint64_t threshold = -n % n;
    for (;;) {
      uint64_t x = next();
      if (x >= threshold)
        return x % n;
    }
  }
};
struct BStream {
  int n;
  BRng r;
  std::vector<int32_t> perm;
  BStream(int size, uint64_t seed) : n(size), r(seed + 23000000), perm(size) {
    for (int i = 0; i < n; ++i)
      perm[i] = i;
  }
};
extern "C" {
void *fb_create(int n, uint64_t seed) {
  if (n <= 0)
    return nullptr;
  return new BStream(n, seed);
}
void fb_destroy(void *q) { delete (BStream *)q; }
int fb_state_size(void *q) { return 32 + 4 * ((BStream *)q)->n; }
void fb_save(void *q, void *out) {
  auto &a = *(BStream *)q;
  std::memcpy(out, a.r.s, 32);
  std::memcpy((char *)out + 32, a.perm.data(), 4 * a.n);
}
void fb_load(void *q, const void *in) {
  auto &a = *(BStream *)q;
  std::memcpy(a.r.s, in, 32);
  std::memcpy(a.perm.data(), (char *)in + 32, 4 * a.n);
}
int fb_draw(void *q, const float *x, const int64_t *y, int b, float *ox,
            int64_t *oy, int64_t *ids, int threads) {
  auto &a = *(BStream *)q;
  if (b < 1 || b > a.n || threads < 1 || threads > 64)
    return 1;
  for (int i = 0; i < b; ++i) {
    int j = i + a.r.bounded(a.n - i);
    std::swap(a.perm[i], a.perm[j]);
    ids[i] = a.perm[i];
  }
#pragma omp parallel for num_threads(threads) if (b >= 512) schedule(static)
  for (int i = 0; i < b; ++i) {
    std::memcpy(ox + i * 49, x + ids[i] * 49, 49 * sizeof(float));
    oy[i] = y[ids[i]];
  }
  return 0;
}
}
