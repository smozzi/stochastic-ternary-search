#include <cstdint>
extern "C" int sparse3_scores(const float *, const int64_t *, int,
                              const int8_t *, const int *, const int8_t *, int,
                              int, int, const float *, const float *, int,
                              double *);
extern "C" int sparse3_512_scores(const float *, const int64_t *, int,
                                  const int8_t *, const int *, const int8_t *,
                                  int, int, int, const float *, const float *,
                                  int, double *);
extern "C" int adaptive_sparse_scores(const float *x, const int64_t *y, int n,
                                      const int8_t *b, const int *i,
                                      const int8_t *v, int d, int p, int h,
                                      const float *delta, const float *tau,
                                      int threads, double *out) {
  static const bool supported = __builtin_cpu_supports("avx512f");
  auto fn =
      supported && n / threads >= 16 ? sparse3_512_scores : sparse3_scores;
  return fn(x, y, n, b, i, v, d, p, h, delta, tau, threads, out);
}
