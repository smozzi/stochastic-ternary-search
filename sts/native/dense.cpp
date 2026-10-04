// AVX2 across eight images: bounded tile memory, same sum order, vector
// exp/log.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <vector>
extern "C" __m256 _ZGVdN8v_expf(__m256);
extern "C" __m256 _ZGVdN8v_logf(__m256);
struct SPatch {
  std::vector<std::pair<int, int>> w1, w2;
  std::vector<int> hidden;
};
static __m256 sgain(int8_t q) {
  return _mm256_set1_ps(q < 0 ? .5f : q == 0 ? 1.f : 2.f);
}
static __m256 activate(__m256 z, float tau, int8_t g) {
  auto u = _mm256_div_ps(z, _mm256_set1_ps(tau));
  return _mm256_mul_ps(_mm256_add_ps(u, _mm256_mul_ps(u, u)), sgain(g));
}
extern "C" int simd_scores(const float *x, const int64_t *y, int n,
                           const int8_t *base, const int8_t *a, int p,
                           const int *sizes, int layers, const float *delta,
                           const float *tau, int threads, int incremental,
                           double *out) {
  if (layers != 2 || sizes[0] != 49 || sizes[2] != 10 || sizes[1] > 128 ||
      sizes[1] < 1 || p < 1 || p > 32 || n < 1 || threads < 1 || threads > 64)
    return 1;
  int h = sizes[1], nw = 59 * h, go = nw, bo = nw + h, total = 61 * h + 10;
  std::vector<SPatch> patch(p);
  for (int c = 0; c < p; ++c) {
    auto v = a + c * total;
    bool affected[128] = {};
    for (int i = 0; i < 49 * h; ++i)
      if (v[i] != base[i]) {
        patch[c].w1.emplace_back(i, int(v[i]) - int(base[i]));
        affected[i / 49] = true;
      }
    for (int j = 0; j < h; ++j)
      affected[j] |= v[go + j] != base[go + j] || v[bo + j] != base[bo + j];
    for (int j = 0; j < h; ++j)
      if (affected[j])
        patch[c].hidden.push_back(j);
    for (int i = 0; i < 10 * h; ++i)
      if (v[49 * h + i] != base[49 * h + i])
        patch[c].w2.emplace_back(i, int(v[49 * h + i]) - int(base[49 * h + i]));
  }
  std::vector<double> sums(threads * p, 0.);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int t = 0; t < threads; ++t) {
    int begin = int(int64_t(n) * t / threads),
        end = int(int64_t(n) * (t + 1) / threads);
    double *acc = sums.data() + t * p;
    alignas(32) float tile[49][8], tmp[8];
    __m256 in[49], zb[128], hb[128], ob[10], zc[128], hc[128], oc[10];
    for (int s = begin; s < end; s += 8) {
      int valid = std::min(8, end - s);
      for (int i = 0; i < 49; ++i) {
        for (int lane = 0; lane < 8; ++lane)
          tile[i][lane] = lane < valid ? x[(s + lane) * 49 + i] : 0.f;
        in[i] = _mm256_load_ps(tile[i]);
      }
      if (incremental) {
        for (int j = 0; j < h; ++j) {
          auto z = _mm256_setzero_ps();
          for (int i = 0; i < 49; ++i)
            z = _mm256_add_ps(
                z,
                _mm256_mul_ps(_mm256_set1_ps(float(base[j * 49 + i])), in[i]));
          zb[j] =
              _mm256_add_ps(z, _mm256_set1_ps(delta[j] * float(base[bo + j])));
          hb[j] = activate(zb[j], tau[j], base[go + j]);
        }
        for (int k = 0; k < 10; ++k) {
          auto z = _mm256_setzero_ps();
          for (int j = 0; j < h; ++j)
            z = _mm256_add_ps(
                z, _mm256_mul_ps(
                       _mm256_set1_ps(float(base[49 * h + k * h + j])), hb[j]));
          ob[k] = _mm256_add_ps(
              z, _mm256_set1_ps(delta[h + k] * float(base[bo + h + k])));
        }
      }
      for (int c = 0; c < p; ++c) {
        auto v = a + c * total;
        const auto &pt = patch[c];
        if (incremental) {
          for (int j = 0; j < h; ++j) {
            zc[j] = zb[j];
            hc[j] = hb[j];
          }
          for (auto d : pt.w1)
            zc[d.first / 49] = _mm256_add_ps(
                zc[d.first / 49], _mm256_mul_ps(_mm256_set1_ps(float(d.second)),
                                                in[d.first % 49]));
          for (int j : pt.hidden) {
            zc[j] = _mm256_add_ps(
                zc[j], _mm256_set1_ps(delta[j] * float(int(v[bo + j]) -
                                                       int(base[bo + j]))));
            hc[j] = activate(zc[j], tau[j], v[go + j]);
          }
          for (int k = 0; k < 10; ++k)
            oc[k] = ob[k];
          for (int j : pt.hidden) {
            auto dh = _mm256_sub_ps(hc[j], hb[j]);
            if (_mm256_movemask_ps(
                    _mm256_cmp_ps(dh, _mm256_setzero_ps(), _CMP_NEQ_OQ)) == 0)
              continue;
            for (int k = 0; k < 10; ++k)
              oc[k] = _mm256_add_ps(
                  oc[k],
                  _mm256_mul_ps(_mm256_set1_ps(float(base[49 * h + k * h + j])),
                                dh));
          }
          for (auto d : pt.w2)
            oc[d.first / h] = _mm256_add_ps(
                oc[d.first / h], _mm256_mul_ps(_mm256_set1_ps(float(d.second)),
                                               hc[d.first % h]));
          for (int k = 0; k < 10; ++k)
            oc[k] = _mm256_div_ps(
                _mm256_add_ps(oc[k],
                              _mm256_set1_ps(delta[h + k] *
                                             float(int(v[bo + h + k]) -
                                                   int(base[bo + h + k])))),
                _mm256_set1_ps(tau[h + k]));
        } else {
          for (int j = 0; j < h; ++j) {
            auto z = _mm256_setzero_ps();
            for (int i = 0; i < 49; ++i)
              z = _mm256_add_ps(
                  z,
                  _mm256_mul_ps(_mm256_set1_ps(float(v[j * 49 + i])), in[i]));
            z = _mm256_add_ps(z, _mm256_set1_ps(delta[j] * float(v[bo + j])));
            hc[j] = activate(z, tau[j], v[go + j]);
          }
          for (int k = 0; k < 10; ++k) {
            auto z = _mm256_setzero_ps();
            for (int j = 0; j < h; ++j)
              z = _mm256_add_ps(
                  z, _mm256_mul_ps(_mm256_set1_ps(float(v[49 * h + k * h + j])),
                                   hc[j]));
            oc[k] = _mm256_div_ps(
                _mm256_add_ps(
                    z, _mm256_set1_ps(delta[h + k] * float(v[bo + h + k]))),
                _mm256_set1_ps(tau[h + k]));
          }
        }
        auto mx = oc[0];
        for (int k = 1; k < 10; ++k)
          mx = _mm256_max_ps(mx, oc[k]);
        auto sum = _mm256_setzero_ps();
        auto target = _mm256_setzero_ps();
        for (int k = 0; k < 10; ++k) {
          auto centered = _mm256_sub_ps(oc[k], mx);
          sum = _mm256_add_ps(sum, _ZGVdN8v_expf(centered));
          alignas(32) int mask[8];
          for (int l = 0; l < 8; ++l)
            mask[l] = (l < valid && y[s + l] == k) ? -1 : 0;
          target = _mm256_blendv_ps(
              target, centered,
              _mm256_castsi256_ps(_mm256_load_si256((__m256i *)mask)));
        }
        auto losses = _mm256_sub_ps(_ZGVdN8v_logf(sum), target);
        _mm256_store_ps(tmp, losses);
        for (int l = 0; l < valid; ++l)
          acc[c] += double(tmp[l]);
      }
    }
  }
  for (int c = 0; c < p; ++c) {
    double z = 0.;
    for (int t = 0; t < threads; ++t)
      z += sums[t * p + c];
    out[c] = -z / n;
  }
  return 0;
}
