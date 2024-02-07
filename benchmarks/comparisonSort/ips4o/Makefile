include common/seqDefs
CFLAGS = -O3 -DNDEBUG -DINPLACE -DPARLAY_OPENMP -std=c++17  -march=native  -fopenmp=libomp -pthread -latomic -DPARLAY_USE_STD_ALLOC
LFLAGS = -ldl  /usr/lib/x86_64-linux-gnu/libtbb.so.2 -latomic -mcx16 $(JEMALLOC)
CC = clang++

BENCH = sort

include common/MakeBench
