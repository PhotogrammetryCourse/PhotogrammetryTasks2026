#pragma once


#define NO_DEPTH                    0.0f
#define NO_COST                     1.0f
#define GOOD_COST                   0.2f

#define NITERATIONS                 5

#define PROPAGATION_STEP            17

#define COST_PATCH_RADIUS           5

#define COSTS_K_RATIO               1.2f
#define COSTS_BEST_K_LIMIT          5

#define VERBOSE_LOGGING
#ifdef VERBOSE_LOGGING
	#define verbose_cout std::cout
#else
	#define verbose_cout if (true) {} else std::cout
#endif

#define DEBUG_DIR                   "data/debug/test_depth_maps_pm/iterations_points/"

// 208: что если попробовать другой PROPAGATION_STEP?
// Чем больше PROPAGATION_STEP, тем больше разлетаются облака (появляются лучи, улетающие на край сцены).
// Чем меньше, тем плотнее облака.
// В целом 25 как будто бы много, лучше всё-таки чуть меньше
// 209: что если попробовать в PROPAGATION брать 8 из 20 лучших с точки зрения их cost-ов, и уже затем выбирать из них лучшего с учетом примерки на себя?
// Как будто бы ничего не меняется. Ну и в целом из групп в прошлом варианте как будто бы нет смысла брать больше 1 гипотезы,
// а такая модификация будет заниматься именно этим, если увидит плохую группу (из которой ничего не возьмёт)
