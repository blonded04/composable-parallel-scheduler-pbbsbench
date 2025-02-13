// Scheduler plugin for OpenMP
//
// Critical Note: It is **very important** that we do not accidentally nest
// parallel regions in OpenMP, because this will result in duplicate worker
// IDs (each team gets assigned their own sequential worker IDs from 0 to
// omp_get_num_threads() - 1). Therefore, we always check whether we are
// already inside a parallel region before creating one. If we are already
// inside one, tasks will just make use of the existing threads in the team.
//

#ifndef PARLAY_INTERNAL_SCHEDULER_PLUGINS_OMP_H_
#define PARLAY_INTERNAL_SCHEDULER_PLUGINS_OMP_H_

#if !defined(PARLAY_OPENMP) || !defined(OMP_MODE)
#error "Undefined omp"
#endif

#include <omp.h>

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <cassert>

#include "common/initialization.h"
#include "eigen/util.h"

namespace parlay {

// IWYU pragma: private, include "../../parallel.h"

inline size_t num_workers() {
  return omp_get_max_threads();
}

inline size_t worker_id() {
  return omp_get_thread_num();
}


template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long granularity, bool) {
  static_assert(std::is_invocable_v<F&, size_t>);

  if (end == start + 1) {
    f(start);
  }
  else if ((end - start) <= static_cast<size_t>(granularity)) {
    for (size_t i=start; i < end; i++) {
      f(i);
    }
  }
  // else if (!omp_in_parallel()) {
  //   throw std::runtime_error{"!omp_in_parallel()"};
  //   #pragma omp parallel
  //   {
  //     #pragma omp single
  //     {
  //       if (granularity <= 1) {
  //         #pragma omp taskloop
  //         for (size_t i = start; i < end; i++) {
  //           f(i);
  //         }
  //       }
  //       else {
  //         #pragma omp taskloop grainsize(granularity)
  //         for (size_t i = start; i < end; i++) {
  //           f(i);
  //         }
  //       }
  //     }
  //   }
  // }
  else {
    // #pragma omp taskloop shared(f)
    #pragma omp parallel
    #if OMP_MODE == OMP_STATIC
    #pragma omp for schedule(static)
    #elif OMP_MODE == OMP_RUNTIME
    #pragma omp for schedule(runtime)
    #elif OMP_MODE == OMP_DYNAMIC_MONOTONIC
    // TODO: chunk size?
    #pragma omp for schedule(monotonic : dynamic)
    #elif OMP_MODE == OMP_DYNAMIC_NONMONOTONIC
    #pragma omp for schedule(nonmonotonic : dynamic)
    #elif OMP_MODE == OMP_GUIDED_MONOTONIC
    #pragma omp for schedule(monotonic : guided)
    #elif OMP_MODE == OMP_GUIDED_NONMONOTONIC
    #pragma omp for schedule(nonmonotonic : guided)
    #else
      static_assert(false, "Wrong OMP_MODE mode");
    #endif
    for (size_t i = start; i < end; i++) {
      f(i);
    }
  }
}

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool) {
  static_assert(std::is_invocable_v<Lf&&>);
  static_assert(std::is_invocable_v<Rf&&>);

  // If we are not yet in a parallel region, start one
  if (!omp_in_parallel()) {
    #pragma omp parallel
    {
      #pragma omp single
      {
        #pragma omp taskgroup
        {
          #pragma omp task untied
          { std::forward<Rf>(right)(); }

          { std::forward<Lf>(left)(); }
        }
      }  // omp single
    }  // omp parallel
  }
  // Already inside a parallel region, avoid creating nested one (see comment at top)
  else {
    #pragma omp taskgroup
    {
      #pragma omp task untied shared(right)
      { std::forward<Rf>(right)(); }

      { std::forward<Lf>(left)(); }
    }
  }
}

inline void init_plugin_internal() {
  auto threadsNum = num_workers();
  static internal::InitOnce ompInit{[threadsNum] { omp_set_num_threads(static_cast<int>(threadsNum)); }};
}

template <typename... Fs>
void execute_with_scheduler(Fs...) {
  struct Illegal {};
  static_assert((std::is_same_v<Illegal, Fs> && ...), "parlay::execute_with_scheduler is only available in the Parlay scheduler and is not compatible with OpenMP");
}

}  // namespace parlay

#endif  // PARLAY_INTERNAL_SCHEDULER_PLUGINS_OMP_H_
