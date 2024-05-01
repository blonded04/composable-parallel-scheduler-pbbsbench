#ifndef PARLAY_INTERNAL_SCHEDULER_PLUGINS_EIGEN_H_
#define PARLAY_INTERNAL_SCHEDULER_PLUGINS_EIGEN_H_

#include <thread>

#include "eigen/eigen_pinner.h"
#include "eigen/poor_barrier.h"
#include "eigen/thread_index.h"
#include "eigen/timespan_partitioner.h"
#include "eigen/util.h"

namespace parlay {

namespace internal {
// TODO: move to eigen header
template <typename F>
inline void EigenParallelFor(size_t from, size_t to, F &&func, long grain_size) {
#if EIGEN_MODE == EIGEN_SIMPLE
  Eigen::Partitioner::ParallelForSimple(from, to, std::forward<F>(func), grain_size);
#elif EIGEN_MODE == EIGEN_TIMESPAN
  Eigen::Partitioner::ParallelForTimespan<Eigen::Partitioner::GrainSize::DEFAULT>(
      from, to, std::forward<F>(func), grain_size);
#elif EIGEN_MODE == EIGEN_TIMESPAN_GRAINSIZE
  Eigen::Partitioner::ParallelForTimespan<Eigen::Partitioner::GrainSize::AUTO>(
      from, to, std::forward<F>(func), grain_size);
#elif EIGEN_MODE == EIGEN_STATIC
  Eigen::Partitioner::ParallelForStatic(from, to, std::forward<F>(func), grain_size);
#elif EIGEN_MODE == EIGEN_RAPID
  RapidGroup.parallel_ranges(from, to, [&func](auto from, auto to, auto part) {
    for (size_t i = from; i != to; ++i) {
      func(i);
    }
  });
#else
  static_assert(false, "Wrong EIGEN_MODE mode");
#endif
}

}

inline size_t num_workers() {
  // cache result to avoid calling getenv on every call
  static size_t threads = []() -> size_t {
    if (const char *envThreads = std::getenv("BENCH_NUM_THREADS")) {
      return std::stoul(envThreads);
    }
    // left just for compatibility
    if (const char *envThreads = std::getenv("BENCH_MAX_THREADS")) {
      return std::stoul(envThreads);
    }
    return std::thread::hardware_concurrency();
  }();
  return threads;
}

inline size_t worker_id() {
    return GetThreadIndex();
}

template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long grain_size, bool) {
  internal::EigenParallelFor(start, end, std::forward<F>(f), grain_size);
}

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool) {
  Eigen::Partitioner::ParallelDo(std::forward<Lf>(left), std::forward<Rf>(right));
}

inline void init_plugin_internal() {
    auto threadsNum = num_workers();

    #if defined(EIGEN_MODE) and EIGEN_MODE != EIGEN_RAPID
    static EigenPinner pinner(threadsNum);
    #endif
}

template <typename... Fs>
void execute_with_scheduler(Fs...) {
  struct Illegal {};
  static_assert((std::is_same_v<Illegal, Fs> && ...), "parlay::execute_with_scheduler is only available in the Parlay scheduler and is not compatible with OpenMP");
}

}
#endif