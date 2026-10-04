#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
struct RNG {
  uint64_t s[4];
  static uint64_t rot(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  explicit RNG(uint64_t seed) {
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
  double uniform() { return (next() >> 11) * 0x1.0p-53; }
  int trit() { return int(uniform() * 3) - 1; }
};
struct Fenwick {
  int n;
  std::vector<double> q, t;
  explicit Fenwick(int count) : n(count), q(count, 1.), t(count + 1, 0.) {
    for (int i = 0; i < n; ++i)
      add(i, 1.);
  }
  void add(int i, double d) {
    for (++i; i <= n; i += i & -i)
      t[i] += d;
  }
  void set(int i, double v) {
    double d = v - q[i];
    q[i] = v;
    add(i, d);
  }
  double total() const {
    double a = 0.;
    for (int i = n; i; i -= i & -i)
      a += t[i];
    return a;
  }
  int find(double u) const {
    int i = 0;
    for (int b = 1 << int(std::log2(n)); b; b >>= 1) {
      int j = i + b;
      if (j <= n && t[j] <= u) {
        i = j;
        u -= t[j];
      }
    }
    return std::min(i, n - 1);
  }
};
