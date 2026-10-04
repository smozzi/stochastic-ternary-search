#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
compiler=${CXX:-g++}
flags=(-O3 -std=c++17 -mavx2 -fPIC -fopenmp -ffp-contract=off)
for source in search score_avx2 dispatch dense; do
    "$compiler" "${flags[@]}" -c "sts/native/$source.cpp" -o "$work/$source.o"
done
"$compiler" -O3 -std=c++17 -mavx512f -fPIC -fopenmp -ffp-contract=off -c sts/native/score_avx512.cpp -o "$work/score_avx512.o"
"$compiler" -shared -fopenmp "$work"/*.o -o sts/_search.so -lmvec -lm
"$compiler" -O3 -std=c++17 -fPIC -shared -fopenmp sts/native/batch.cpp -o sts/_batch.so
"$compiler" "${flags[@]}" -shared sts/native/probe.cpp -o sts/_probe.so -lmvec -lm
