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
struct SPatch3 {
  std::vector<std::pair<int, int>> w1, w2;
  std::vector<int> hidden;
  int8_t gains[128];
  float bias[128] = {}, outbias[10] = {};
  bool affected[128] = {};
};
static __m256 sgain(int8_t q) {
  return _mm256_set1_ps(q < 0 ? .5f : q == 0 ? 1.f : 2.f);
}
static __m256 activate(__m256 z, float tau, int8_t g) {
  auto u = _mm256_div_ps(z, _mm256_set1_ps(tau));
  return _mm256_mul_ps(_mm256_add_ps(u, _mm256_mul_ps(u, u)), sgain(g));
}
extern "C" int sparse3_scores(const float *x, const int64_t *y, int n,
                              const int8_t *base, const int *indices,
                              const int8_t *values, int d, int p, int hidden,
                              const float *delta, const float *tau, int threads,
                              double *out) {
  int sizes[3] = {49, hidden, 10}, layers = 2, incremental = 1;
  if (layers != 2 || sizes[0] != 49 || sizes[2] != 10 || sizes[1] > 128 ||
      sizes[1] < 1 || p < 1 || p > 32 || n < 1 || threads < 1 || threads > 64)
    return 1;
  int h = sizes[1], nw = 59 * h, go = nw, bo = nw + h, total = 61 * h + 10;
  std::vector<SPatch3> patch(p);
  for (int c = 0; c < p; ++c) {
    auto &pt = patch[c];
    std::copy(base + go, base + go + h, pt.gains);
    for (int z = 0; z < d; ++z) {
      int i = indices[z], diff = int(values[c * d + z]) - int(base[i]);
      if (!diff)
        continue;
      if (i < 49 * h) {
        pt.w1.emplace_back(i, diff);
        pt.affected[i / 49] = true;
      } else if (i < nw)
        pt.w2.emplace_back(i - 49 * h, diff);
      else if (i < bo) {
        pt.gains[i - go] = values[c * d + z];
        pt.affected[i - go] = true;
      } else if (i < bo + h) {
        pt.bias[i - bo] = delta[i - bo] * float(diff);
        pt.affected[i - bo] = true;
      } else
        pt.outbias[i - bo - h] = delta[i - bo] * float(diff);
    }
    for (int j = 0; j < h; ++j)
      if (pt.affected[j])
        pt.hidden.push_back(j);
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
      alignas(32) int masks[10][8] = {};
      for (int l = 0; l < valid; ++l) {
        int label = int(y[s + l]);
        if (label >= 0 && label < 10)
          masks[label][l] = -1;
      }
      __m256 targets[10];
      for (int k = 0; k < 10; ++k)
        targets[k] =
            _mm256_castsi256_ps(_mm256_load_si256((__m256i *)masks[k]));
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
        auto v = base;
        const auto &pt = patch[c];
        if (incremental) {
          for (int j : pt.hidden)
            zc[j] = zb[j];
          for (auto d : pt.w1)
            zc[d.first / 49] = _mm256_add_ps(
                zc[d.first / 49], _mm256_mul_ps(_mm256_set1_ps(float(d.second)),
                                                in[d.first % 49]));
          for (int j : pt.hidden) {
            zc[j] = _mm256_add_ps(zc[j], _mm256_set1_ps(pt.bias[j]));
            hc[j] = activate(zc[j], tau[j], pt.gains[j]);
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
                oc[d.first / h],
                _mm256_mul_ps(_mm256_set1_ps(float(d.second)),
                              (pt.affected[d.first % h] ? hc[d.first % h]
                                                        : hb[d.first % h])));
          for (int k = 0; k < 10; ++k)
            oc[k] = _mm256_div_ps(
                _mm256_add_ps(oc[k], _mm256_set1_ps(pt.outbias[k])),
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
          target = _mm256_blendv_ps(target, centered, targets[k]);
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
