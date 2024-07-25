#ifndef PARLAY_INTERNAL_SCHEDULER_PLUGINS_TASKFLOW_H_
#define PARLAY_INTERNAL_SCHEDULER_PLUGINS_TASKFLOW_H_

#include "taskflow/taskflow/taskflow.hpp"
#include "taskflow/taskflow/algorithm/for_each.hpp"
#include "eigen/modes.h"

namespace parlay {

static inline tf::Executor exec;

inline size_t num_workers() {
    return exec.num_workers();
}

inline size_t worker_id() {
    return exec.this_worker_id();
}

template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long granularity, bool) {
    tf::Taskflow tf;

#if TASKFLOW_MODE == TASKFLOW_GUIDED
    tf::ExecutionPolicy<tf::GuidedPartitioner> execution_policy;
#elif TASKFLOW_MODE == TASKFLOW_DYNAMIC
    tf::ExecutionPolicy<tf::DynamicPartitioner> execution_policy;
#elif TASKFLOW_MODE == TASKFLOW_STATIC
    tf::ExecutionPolicy<tf::StaticPartitioner> execution_policy;
#elif TASKFLOW_MODE == TASKFLOW_RANDOM
    tf::ExecutionPolicy<tf::RandomPartitioner> execution_policy;
#else
    static_assert(false, "Wrong TASKFLOW_MODE mode");
#endif  // TASKFLOW_MODE
    
    tf.for_each_index(start, end, granularity, std::forward<F>(f), execution_policy);

    exec.run(taskflow).get();
}

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool) {
    tf::Taskflow tf;

    tf.emplace(std::forward<Lf>(left));
    auto fut = exec.run(taskflow);

    std::forward<Rf>(right)();

    fut.get();
}

inline void init_plugin_internal() {}

template <typename... Fs>
void execute_with_scheduler(Fs...) {
    struct Illegal {};
    static_assert((std::is_same_v<Illegal, Fs> && ...), "parlay::execute_with_scheduler is only available in the Parlay scheduler and is not compatible with OpenMP");
}

}  // namespace parlay

#endif  // PARLAY_INTERNAL_SCHEDULER_PLUGINS_TASKFLOW_H_
