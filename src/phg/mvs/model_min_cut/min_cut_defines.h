// Над saharov32 на тесте FromAllDepthMaps были проведены следующие эксперименты с MERGE_THRESHOLD_RADIUS_KOEF.
//
// (здесь число real vertices берётся перед добавлением bbox-а, mesh - из итогового ply)
// koef | real vertices | mesh vertices/faces
// 0.03 |         86805 |        85689 / 206304
// 0.10 |         86291 |        85174 / 204964
// 0.20 |         83031 |        81915 / 196660
// 0.40 |         68102 |        67014 / 159116
//
// Визуально прям сильной разницы я почти не заметил. Разве что при значении 0.4 становится видно,
// что поверхности действительно начинают становиться грубее и что склеиваются соседние разные участки поверхности.
// Оставил 0.1, ибо всё-таки не хочется терять никаких деталей даже за счёт потенциального шума,
// но хоть какие-то, самые близкорасположенные, точки всё же хочется склеить.
#define MERGE_THRESHOLD_RADIUS_KOEF     0.1

#define LAMBDA_OUT                      1.0
#define LAMBDA_IN                       1.0

#define MIN_CUT_VISIBILITY_SIGMA_RADIUS_KOEF     1.0
#define MIN_CUT_SINK_DEPTH_SIGMA_KOEF            3.0

#define MIN_CUT_AVERAGE_MERGED_VERTEX_COLORS     0

// Над saharov32 на тесте FromAllDepthMaps были проведены следующие замеры, проверяющие, как ускорился код.
//
// - MIN_CUT_ENABLE_PARALLEL_RAY_TRACING=0 и MIN_CUT_USE_FAST_FLOAT_INTERSECTION=0:
//   ray traversal 97.7479 s, buildMesh 104.658 s, mesh 85174 vertices / 204964 faces.
// - MIN_CUT_ENABLE_PARALLEL_RAY_TRACING=0 и MIN_CUT_USE_FAST_FLOAT_INTERSECTION=1:
//   ray traversal 10.9155 s, buildMesh 17.9592 s, mesh 85174 vertices / 204964 faces.
// - MIN_CUT_ENABLE_PARALLEL_RAY_TRACING=1 и MIN_CUT_USE_FAST_FLOAT_INTERSECTION=0:
//   ray traversal 17.0942 s, buildMesh 24.1728 s, mesh 85174 vertices / 204964 faces.
// - MIN_CUT_ENABLE_PARALLEL_RAY_TRACING=1 и MIN_CUT_USE_FAST_FLOAT_INTERSECTION=1:
//   ray traversal 2.42188 s, buildMesh 9.26079 s, mesh 85174 vertices / 204964 faces.
//
// Значит, мы за бесплатно получили ускорение в ~40 раз по трассировке лучей.
// Здесь хочется вставить эту гифку: https://t.me/c/3808617323/387
#define MIN_CUT_ENABLE_PARALLEL_RAY_TRACING      1

#define MIN_CUT_USE_FAST_FLOAT_INTERSECTION      1
#define MIN_CUT_FAST_INTERSECTION_EXACT_FALLBACK 1
