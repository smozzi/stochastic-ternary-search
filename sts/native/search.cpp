#define sparse_scores adaptive_sparse_scores
// Store only selected coordinates of candidates, retaining native RNG/score
// laws.
#include "native_sampling.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
extern "C" int simd_scores(const float *, const int64_t *, int, const int8_t *,
                           const int8_t *, int, const int *, int, const float *,
                           const float *, int, int, double *);
extern "C" int sparse_scores(const float *, const int64_t *, int,
                             const int8_t *, const int *, const int8_t *, int,
                             int, int, const float *, const float *, int,
                             double *);
struct SparseRCT {
  int h, n, nw, ng, nb;
  int8_t *leader;
  void *e, *mass;
  int precision;
  double decay;
  std::unordered_map<int, std::pair<int64_t, int64_t>> extension;
  RNG wr, gr, br;
  Fenwick tree;
  std::vector<double> gpref;
  std::vector<int> idx;
  std::vector<int8_t> candidates, group, proposal, old, dense;
  SparseRCT(int hidden, int8_t *l, void *ee, void *mm, uint64_t seed, int pp,
            double dd)
      : h(hidden), n(61 * h + 10), nw(59 * h), ng(h), nb(h + 10), leader(l),
        e(ee), mass(mm), precision(pp), decay(dd), wr(seed + 17000000),
        gr(seed + 19000000), br(seed + 21000000), tree(nw), gpref(ng + nb) {
    for (int i = 0; i < nw; ++i)
      tree.set(i, pref(i));
    for (int i = nw; i < n; ++i)
      gpref[i - nw] = pref(i);
  }
  std::pair<int64_t, int64_t> counts(int i) const {
    auto it = extension.find(i);
    return it == extension.end() ? std::make_pair(int64_t(((int32_t *)e)[i]),
                                                  int64_t(((int32_t *)mass)[i]))
                                 : it->second;
  }
  void set_counts(int i, int64_t a, int64_t b) {
    if (b < 0)
      throw std::runtime_error("negative mass");
    if (a >= INT32_MIN && a <= INT32_MAX && b <= INT32_MAX) {
      extension.erase(i);
      ((int32_t *)e)[i] = int32_t(a);
      ((int32_t *)mass)[i] = int32_t(b);
    } else {
      extension[i] = {a, b};
      ((int32_t *)e)[i] = ((int32_t *)mass)[i] = 0;
    }
  }
  double get(void *p, int i) const {
    return precision == 64   ? ((double *)p)[i]
           : precision == 32 ? ((float *)p)[i]
                             : float(((_Float16 *)p)[i]);
  }
  void put(void *p, int i, double value) {
    if (precision == 64)
      ((double *)p)[i] = value;
    else if (precision == 32)
      ((float *)p)[i] = float(value);
    else
      ((_Float16 *)p)[i] = (_Float16)float(value);
  }
  void update_conf(int i, int inc, int cnt, bool changed) {
    if (precision == 31) {
      if (changed) {
        set_counts(i, 0, 0);
        return;
      }
      auto c = counts(i);
      __int128 a = (__int128)c.first + inc,
               b = (__int128)c.second + cnt * (24 - cnt);
      if (a < INT64_MIN || a > INT64_MAX || b > INT64_MAX)
        throw std::overflow_error("confidence INT64 capacity exhausted");
      set_counts(i, int64_t(a), int64_t(b));
      return;
    }
    if (precision == 0)
      return;
    if (changed) {
      put(e, i, 0.);
      put(mass, i, 0.);
    } else if (precision == 64) {
      put(e, i, decay * get(e, i) + inc / 24.);
      put(mass, i, decay * get(mass, i) + cnt * (24 - cnt) / 24.);
    } else {
      float de = float(decay) * float(get(e, i)),
            dm = float(decay) * float(get(mass, i));
      float ne = de + float(inc) / 24.f,
            nm = dm + float(cnt * (24 - cnt)) / 24.f;
      put(e, i, ne);
      put(mass, i, nm);
    }
  }
  double pref(int i) const {
    if (precision == 31) {
      auto c = counts(i);
      double ratio =
          double(c.first) / std::sqrt(24. * (double(c.second) + 72.));
      return .001 + .999 * std::exp(-std::max(-3., std::min(3., ratio)));
    }
    if (precision == 0)
      return 1.;
    double a;
    if (precision == 64)
      a = std::max(-3., std::min(3., get(e, i) / std::sqrt(get(mass, i) + 3.)));
    else {
      float den = std::sqrt(float(get(mass, i)) + 3.f);
      float ratio = float(get(e, i)) / den;
      a = std::max(-3.f, std::min(3.f, ratio));
    }
    return .001 + .999 * std::exp(-a);
  }
  void sample(int k) {
    idx.clear();
    std::vector<double> removed;
    removed.reserve(k);
    for (int z = 0; z < k; ++z) {
      double total = tree.total();
      int i =
          tree.find(std::min(std::nextafter(total, 0.), wr.uniform() * total));
      if (tree.q[i] <= 0)
        throw std::runtime_error("zero interval");
      idx.push_back(i);
      removed.push_back(tree.q[i]);
      tree.set(i, 0.);
    }
    for (int z = 0; z < k; ++z)
      tree.set(idx[z], removed[z]);
    std::sort(idx.begin(), idx.end());
    for (int g = 0; g < 2; ++g) {
      int begin = g ? nw + ng : nw, count = g ? nb : ng;
      RNG &r = g ? br : gr;
      double sum = 0.;
      for (int i = 0; i < count; ++i)
        sum += gpref[begin + i - nw];
      double budget = count * 4. / nw;
      for (int i = 0; i < count; ++i)
        if (r.uniform() < gpref[begin + i - nw] * (budget / sum))
          idx.push_back(begin + i);
    }
    int d = idx.size();
    candidates.resize(24 * d);
    group.resize(25 * d);
    proposal.resize(d);
    old.resize(d);
    for (int z = 0; z < d; ++z)
      old[z] = leader[idx[z]];
    for (int c = 0; c < 24; ++c)
      for (int z = 0; z < d; ++z) {
        int i = idx[z];
        candidates[c * d + z] =
            i < nw ? wr.trit() : (i < nw + ng ? gr.trit() : br.trit());
      }
  }
  bool equal(const int8_t *a, const int8_t *b, int d) const {
    return std::memcmp(a, b, d) == 0;
  }
  int score(const float *x, const int64_t *y, int b, const float *delta,
            const float *tau, int threads, const int8_t *values, int p,
            bool inc, double *out) {
    int d = idx.size();
    if (inc)
      return sparse_scores(x, y, b, leader, idx.data(), values, d, p, h, delta,
                           tau, threads, out);
    dense.resize(p * n);
    for (int c = 0; c < p; ++c) {
      std::memcpy(dense.data() + c * n, leader, n);
      for (int z = 0; z < d; ++z)
        dense[c * n + idx[z]] = values[c * d + z];
    }
    int sizes[3] = {49, h, 10};
    return simd_scores(x, y, b, leader, dense.data(), p, sizes, 2, delta, tau,
                       threads, 0, out);
  }
  int step(const float *x, const int64_t *y, int b, const float *delta,
           const float *tau, int threads, int k, double *out) {
    if (b < 1)
      return 12;
    for (int s = 0; s < b; ++s)
      if (y[s] < 0 || y[s] >= 10)
        return 12;
    sample(k);
    int d = idx.size(), active = 0;
    std::memcpy(group.data(), old.data(), d);
    int map[24];
    for (int c = 0; c < 24; ++c) {
      int j = 0;
      for (; j <= active; ++j)
        if (equal(candidates.data() + c * d, group.data() + j * d, d))
          break;
      if (j > active) {
        ++active;
        std::memcpy(group.data() + active * d, candidates.data() + c * d, d);
      }
      map[c] = j;
    }
    double unique[25], scores[24];
    int status = score(x, y, b, delta, tau, threads, group.data(), active + 1,
                       active > 1, unique);
    if (status)
      return status;
    for (int j = 0; j <= active; ++j)
      if (!std::isfinite(unique[j]))
        return 8;
    double before = unique[0];
    int improvers = 0;
    for (int c = 0; c < 24; ++c) {
      scores[c] = unique[map[c]];
      improvers += scores[c] > before;
    }
    std::memcpy(proposal.data(), old.data(), d);
    if (k > 1 && active >= 2 && improvers)
      for (int z = 0; z < d; ++z) {
        int value = 2;
        bool agree = true;
        for (int c = 0; c < 24; ++c)
          if (scores[c] > before) {
            int v = candidates[c * d + z];
            if (value == 2)
              value = v;
            else if (value != v) {
              agree = false;
              break;
            }
          }
        if (agree)
          proposal[z] = int8_t(value);
      }
    int j = 0;
    for (; j <= active; ++j)
      if (equal(proposal.data(), group.data() + j * d, d))
        break;
    bool extra = j > active;
    double ps;
    if (extra) {
      status =
          score(x, y, b, delta, tau, threads, proposal.data(), 1, false, &ps);
      if (status || !std::isfinite(ps))
        return 9;
    } else
      ps = unique[j];
    double best = before;
    const int8_t *winner = old.data();
    for (int c = 0; c < 24; ++c)
      if (scores[c] > best) {
        best = scores[c];
        winner = candidates.data() + c * d;
      }
    if (ps > best) {
      best = ps;
      winner = proposal.data();
    }
    int ranks[24], order[24];
    for (int c = 0; c < 24; ++c)
      order[c] = c;
    std::sort(order, order + 24,
              [&](int a, int b) { return scores[a] < scores[b]; });
    for (int l = 0; l < 24;) {
      int r = l + 1;
      while (r < 24 && scores[order[r]] == scores[order[l]])
        ++r;
      for (int z = l; z < r; ++z)
        ranks[order[z]] = l + r - 24;
      l = r;
    }
    int changed = 0, weights = 0;
    for (int z = 0; z < d; ++z) {
      int i = idx[z], cnt = 0, inc = 0;
      for (int c = 0; c < 24; ++c)
        if (candidates[c * d + z] == old[z]) {
          ++cnt;
          inc += ranks[c];
        }
      update_conf(i, inc, cnt, winner[z] != old[z]);
      if (winner[z] != old[z]) {
        ++changed;
        weights += i < nw;
      }
      leader[i] = winner[z];
      if (i < nw)
        tree.set(i, pref(i));
      else
        gpref[i - nw] = pref(i);
    }
    out[0] = before;
    out[1] = best;
    out[2] = std::max(0., best - before);
    out[3] = changed;
    out[4] = weights;
    out[5] = active;
    out[6] = extra;
    out[7] = d;
    return 0;
  }
  size_t base_size() const { return 96 + (2 * nw + 1) * 8; }
  size_t state_size() const {
    return base_size() + (precision == 31 ? 8 + 20 * extension.size() : 0);
  }
  void save(void *buffer) const {
    char *p = (char *)buffer;
    for (const RNG *r : {&wr, &gr, &br}) {
      std::memcpy(p, r->s, 32);
      p += 32;
    }
    std::memcpy(p, tree.q.data(), nw * 8);
    p += nw * 8;
    std::memcpy(p, tree.t.data(), (nw + 1) * 8);
    p += (nw + 1) * 8;
    if (precision == 31) {
      uint64_t count = extension.size();
      std::memcpy(p, &count, 8);
      p += 8;
      std::vector<int> keys;
      for (auto &v : extension)
        keys.push_back(v.first);
      std::sort(keys.begin(), keys.end());
      for (int i : keys) {
        auto c = extension.at(i);
        std::memcpy(p, &i, 4);
        p += 4;
        std::memcpy(p, &c.first, 8);
        p += 8;
        std::memcpy(p, &c.second, 8);
        p += 8;
      }
    }
  }
  void load(const void *buffer) {
    const char *p = (const char *)buffer;
    for (RNG *r : {&wr, &gr, &br}) {
      std::memcpy(r->s, p, 32);
      p += 32;
    }
    std::memcpy(tree.q.data(), p, nw * 8);
    p += nw * 8;
    std::memcpy(tree.t.data(), p, (nw + 1) * 8);
    p += (nw + 1) * 8;
    if (precision == 31) {
      extension.clear();
      uint64_t count;
      std::memcpy(&count, p, 8);
      p += 8;
      for (uint64_t k = 0; k < count; ++k) {
        int i;
        int64_t a, b;
        std::memcpy(&i, p, 4);
        p += 4;
        std::memcpy(&a, p, 8);
        p += 8;
        std::memcpy(&b, p, 8);
        p += 8;
        extension[i] = {a, b};
      }
    }
    for (int i = nw; i < n; ++i)
      gpref[i - nw] = pref(i);
  }
};
extern "C" {
void *nrct_create(int h, int8_t *l, void *e, void *m, uint64_t seed,
                  int precision, double decay) {
  try {
    if (h < 1 || h > 128 ||
        (precision != 31 && precision != 32 && precision != 64) || decay != 1.)
      return nullptr;
    return new SparseRCT(h, l, e, m, seed, precision, decay);
  } catch (...) {
    return nullptr;
  }
}
double nrct_update_conf(void *q, int i, int inc, int cnt, int changed) {
  try {
    auto &a = *(SparseRCT *)q;
    a.update_conf(i, inc, cnt, changed);
    double p = a.pref(i);
    if (i < a.nw)
      a.tree.set(i, p);
    else
      a.gpref[i - a.nw] = p;
    return p;
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}
int nrct_extension_count(void *q) {
  return int(((SparseRCT *)q)->extension.size());
}
int64_t nrct_counter(void *q, int i, int which) {
  auto c = ((SparseRCT *)q)->counts(i);
  return which ? c.second : c.first;
}
void nrct_set_counter(void *q, int i, int64_t e, int64_t m) {
  auto &a = *(SparseRCT *)q;
  a.set_counts(i, e, m);
  double p = a.pref(i);
  if (i < a.nw)
    a.tree.set(i, p);
  else
    a.gpref[i - a.nw] = p;
}
void nrct_destroy(void *q) { delete (SparseRCT *)q; }
void nrct_simd(void *q, int enabled) {}
int nrct_step(void *q, const float *x, const int64_t *y, int b, const float *d,
              const float *t, int threads, int k, double *out) {
  try {
    auto &a = *(SparseRCT *)q;
    if (k < 1 || k > a.nw)
      return 10;
    return a.step(x, y, b, d, t, threads, k, out);
  } catch (...) {
    return 11;
  }
}
int nrct_state_size(void *q) { return ((SparseRCT *)q)->state_size(); }
void nrct_save(void *q, void *b) { ((SparseRCT *)q)->save(b); }
void nrct_load(void *q, const void *b) { ((SparseRCT *)q)->load(b); }
void nrct_draw(void *q, int k, int *out) {
  auto &a = *(SparseRCT *)q;
  a.sample(k);
  std::copy(a.idx.begin(), a.idx.begin() + k, out);
}
}
