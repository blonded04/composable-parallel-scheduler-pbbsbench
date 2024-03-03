#pragma once
#include "modes.h"
// #define EIGEN_MODE EIGEN_TIMESPAN_GRAINSIZE
#include "num_threads.h"

// #ifdef EIGEN_MODE

#define EIGEN_USE_THREADS
#include "nonblocking_thread_pool.h"

#if EIGEN_MODE == EIGEN_RAPID
inline auto EigenPool =
    Eigen::ThreadPool(GetNumThreads() - 1); // 1 for main thread
#else
inline auto EigenPool = Eigen::ThreadPool(Eigen::internal::GetNumThreads(), true,
                                          true); // todo: disable spinning?
#endif

class EigenPoolWrapper {
public:
  template <typename F> void run(F &&f) {
    EigenPool.Schedule(Eigen::MakeTask(std::forward<F>(f)));
  }

  template <typename F> void run_on_thread(F &&f, size_t hint) {
    EigenPool.RunOnThread(Eigen::MakeTask(std::forward<F>(f)), hint);
  }

  bool join_main_thread() { return EigenPool.JoinMainThread(); }

  bool execute_something_else() {
    return EigenPool.TryExecuteSomething();
  }

  void wait() {
    // TODO: implement
  }
};

// #endif
