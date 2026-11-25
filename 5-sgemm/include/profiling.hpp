#pragma once

// Set to true if profiling with Nsight Compute, so the profiler doesn't run more than necessary.
#define PROFILE_WITH_NSIGHT true
#if PROFILE_WITH_NSIGHT == false
uint constexpr runs {100};
#endif