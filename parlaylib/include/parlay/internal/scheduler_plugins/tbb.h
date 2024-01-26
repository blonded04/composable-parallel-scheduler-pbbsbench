#ifndef PARLAY_INTERNAL_SCHEDULER_PLUGINS_TBB_H_
#define PARLAY_INTERNAL_SCHEDULER_PLUGINS_TBB_H_

#include <cstddef>

#include <type_traits>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

#include "common/tbb_pinner.h"
#include "eigen/modes.h"

namespace parlay {

// IWYU pragma: private, include "../../parallel.h"

inline size_t num_workers() { return tbb::this_task_arena::max_concurrency(); }

inline size_t worker_id() {
  auto id = tbb::this_task_arena::current_thread_index();
  return id == tbb::task_arena::not_initialized ? 0 : id;
}

template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long granularity, bool) {
  static_assert(std::is_invocable_v<F&, size_t>);
  // Use TBB's automatic granularity partitioner (tbb::auto_partitioner)
  if (granularity == 0) {
    tbb::parallel_for(tbb::blocked_range<size_t>(start, end), [&](const tbb::blocked_range<size_t>& r) {
      for (auto i = r.begin(); i != r.end(); ++i) {
        f(i);
      }
    }, tbb::auto_partitioner{});
  }
  // Otherwise, use the granularity specified by the user (tbb::simple_partitioner)
  else {
    static tbb::task_group_context context(
      tbb::task_group_context::bound,
      tbb::task_group_context::default_traits |
          tbb::task_group_context::concurrent_wait);
    #if TBB_MODE == TBB_SIMPLE
      const tbb::simple_partitioner part;
    #elif TBB_MODE == TBB_AUTO
      const tbb::auto_partitioner part;
    #elif TBB_MODE == TBB_AFFINITY
      // "it is important that the same affinity_partitioner object be passed to
      // loop templates to be optimized for affinity" see
      // https://spec.oneapi.io/versions/latest/elements/oneTBB/source/algorithms/partitioners/affinity_partitioner.html
      static tbb::affinity_partitioner part;
    #elif TBB_MODE == TBB_CONST_AFFINITY
      tbb::affinity_partitioner part;
    #elif TBB_MODE == TBB_RAPID
      // no partitioner
    #else
      static_assert(false, "Wrong TBB_MODE mode");
    #endif
      // TODO: grain size?
    #if TBB_MODE == TBB_RAPID
      RapidGroup.parallel_ranges(start, end, [&](auto from, auto to, auto part) {
        for (size_t i = from; i != to; ++i) {
          f(i);
        }
      });
    #else
      tbb::parallel_for(
          tbb::blocked_range(start, end),
          [&](const tbb::blocked_range<size_t> &range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
              f(i);
            }
          },
          part, context);
    #endif
  }
}

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool) {
  static_assert(std::is_invocable_v<Lf&&>);
  static_assert(std::is_invocable_v<Rf&&>);
  tbb::parallel_invoke(std::forward<Lf>(left), std::forward<Rf>(right));
}

inline void init_plugin_internal() {
  static PinningObserver pinner; // just init observer
  static tbb::global_control threadLimit(
      tbb::global_control::max_allowed_parallelism, num_workers());
}

template <typename... Fs>
void execute_with_scheduler(Fs...) {
  struct Illegal {};
  static_assert((std::is_same_v<Illegal, Fs> && ...), "parlay::execute_with_scheduler is only available in the Parlay scheduler and is not compatible with TBB");
}

}  // namespace parlay

#endif  // PARLAY_INTERNAL_SCHEDULER_PLUGINS_TBB_H_

