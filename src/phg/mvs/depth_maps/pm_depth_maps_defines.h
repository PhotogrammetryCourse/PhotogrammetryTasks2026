#pragma once


#define NO_DEPTH                    0.0f
#define NO_COST                     1.0f
#define GOOD_COST                   0.2f

#define NITERATIONS                 5

#define COST_PATCH_RADIUS           5

#define COSTS_K_RATIO               1.2f
#define COSTS_BEST_K_LIMIT          5
#define COSTS_ABS_LIMIT             0.5f

#define COST_SUPPORT_BONUS_ENABLED                0
#define COST_SUPPORT_BONUS_MIN_SUPPORTING_CAMERAS 3.0f
#define COST_SUPPORT_BONUS_PER_EXTRA_CAMERA       0.01f
#define COST_SUPPORT_BONUS_MAX                    0.02f

#define VERBOSE_LOGGING
#ifdef VERBOSE_LOGGING
	#define verbose_cout std::cout
#else
	#define verbose_cout if (true) {} else std::cout
#endif

#define DEBUG_DIR                   "data/debug/test_depth_maps_pm/iterations_points/"

// что если попробовать другой PROPAGATION_STEP?

// PROPAGATION_STEP больше не нужен после перехода на ACMH паттерн донорства + логику про "берем 8 лучших по их личной оценке - по их личному cost"

// что если попробовать в PROPAGATION брать 8 из 20 лучших с точки зрения их cost-ов, и уже затем выбирать из них лучшего с учетом примерки на себя?

// аналогичная ситуация, что и выше :)
