#!/bin/bash -x

set -e

if [ "$#" -ne 1 ]; then
    echo "Error: Exactly one argument is required."
    exit 1
fi


# build tbb
cd onetbb
mkdir -p build
mkdir -p ../thd/libs/
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT=OFF
cmake --build build --parallel
cmake --install build --prefix ../thd/libs
cd ../

# run pbbsbench
mkdir -p results
OMP_NUM_THREADS=$1 ./run_benches.py --dir results --eigen --omp --tbb --tbb-path thd/libs
