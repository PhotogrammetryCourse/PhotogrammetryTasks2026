#pragma once


#define NO_DEPTH                    0.0f
#define NO_COST                     1.0f
#define GOOD_COST                   0.2f

#define NITERATIONS                 5

#define PROPAGATION_STEP            25

#define COST_PATCH_RADIUS           5

#define COSTS_K_RATIO               1.2f
#define COSTS_BEST_K_LIMIT          5
#define COSTS_VALUE_LIMIT           0.6f
#define COSTS_CAM_NUM               3

static_assert(COSTS_CAM_NUM <= COSTS_BEST_K_LIMIT);

#define VERBOSE_LOGGING
#ifdef VERBOSE_LOGGING
	#define verbose_cout std::cout
#else
	#define verbose_cout if (true) {} else std::cout
#endif

#define DEBUG_DIR                   "data/debug/test_depth_maps_pm/iterations_points/"
