#include "pm_depth_maps.h"
#include <libutils/fast_random.h>
#include <libutils/timer.h>
#include <phg/utils/point_cloud_export.h>

#include "pm_depth_maps_defines.h"
#include "pm_fast_random.h"
#include "pm_geometry.h"

namespace phg {

matrix3d extractR(const matrix34d& P)
{
    matrix3d RtoLocal;
    vector3d O;
    phg::decomposeUndistortedPMatrix(RtoLocal, O, P);
    return RtoLocal;
}

matrix34d invP(const matrix34d& P)
{
    vector3d p(2.124, 5361.4, 78.6);

    vector3d p01 = P * homogenize(p);

    matrix3d RtoLocal;
    vector3d O;
    phg::decomposeUndistortedPMatrix(RtoLocal, O, P);
    matrix3d RtoWorld = RtoLocal.inv();
    matrix34d Pinv = make34(RtoWorld, O);

    vector3d p10 = Pinv * homogenize(p01);
    rassert(norm2(p10 - p) < 0.00001, 231231241233);

    return Pinv;
}

vector3d project(const vector3d& global_point, const phg::Calibration& calibration, const matrix34d& PtoLocal)
{
    vector3d local_point = PtoLocal * homogenize(global_point);
    double depth = local_point[2];

    vector3f pixel_with_depth = calibration.project(local_point);
    pixel_with_depth[2] = depth; // на самом деле это не глубина, это координата по оси +Z (вдоль которой смотрит камера в ее локальной системе координат)

    return pixel_with_depth;
}

vector3d unproject(const vector3d& pixel, const phg::Calibration& calibration, const matrix34d& PtoWorld)
{
    double depth = pixel[2]; // на самом деле это не глубина, это координата по оси +Z (вдоль которой смотрит камера в ее локальной системе координат)

    vector3d local_point = calibration.unproject(vector2d(pixel[0], pixel[1])) * depth;

    vector3d global_point = PtoWorld * homogenize(local_point);

    return global_point;
}

void PMDepthMapsBuilder::buildDepthMap(unsigned int camera_key, cv::Mat& depth_map_res, cv::Mat& normal_map_res, cv::Mat& cost_map_res, float depth_min, float depth_max)
{
    rassert(camera_key < ncameras, 238192841294108);
    rassert(ncameras >= 2, 21849182491209);

    ref_cam = camera_key;
    ref_depth_min = depth_min;
    ref_depth_max = depth_max;

    width = calibration.width();
    height = calibration.height();

    // в этих трех картинках мы будем хранить для каждого пикселя лучшую на данный момент найденную гипотезу
    depth_map = cv::Mat::zeros(height, width, CV_32FC1); // глубина (точнее координата по оси Z в локальной системе камеры) на которой находится текущая гипотеза (если гипотезы нет - то глубина=0)
    normal_map = cv::Mat::zeros(height, width, CV_32FC3); // нормаль к плоскости поверхности текущей гипотезы (единичный вектор в глобальной системе координат)
    cost_map = cv::Mat::zeros(height, width, CV_32FC1); // оценка качества этой гипотезы (от 0.0 до 1.0, чем меньше - тем лучше гипотеза)

    iter = 0;

    // в первую очередь нам надо заполнить случайными гипотезами, этим займется refinement
    refinement();

    for (iter = 1; iter <= NITERATIONS; ++iter) {
        propagation();
        refinement();
    }

    depth_map_res = depth_map;
    normal_map_res = normal_map;
    cost_map_res = cost_map;
}

void PMDepthMapsBuilder::refinement()
{
    timer t;
    verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": refinement..." << std::endl;

    enum class RefinementCandidateMode
    {
        FiveCandidatesOnly,
        NineCandidatesOnly,
        StagedNineThenFive
    };

    enum RefinementCandidate
    {
        REFINEMENT_KEEP_DEPTH_KEEP_NORMAL,
        REFINEMENT_KEEP_DEPTH_RANDOM_NORMAL,
        REFINEMENT_KEEP_DEPTH_PERTURBED_NORMAL,

        REFINEMENT_RANDOM_DEPTH_KEEP_NORMAL,
        REFINEMENT_RANDOM_DEPTH_RANDOM_NORMAL,
        REFINEMENT_RANDOM_DEPTH_PERTURBED_NORMAL,

        REFINEMENT_PERTURBED_DEPTH_KEEP_NORMAL,
        REFINEMENT_PERTURBED_DEPTH_RANDOM_NORMAL,
        REFINEMENT_PERTURBED_DEPTH_PERTURBED_NORMAL,

        REFINEMENT_NCANDIDATES
    };

    std::array<unsigned long long, REFINEMENT_NCANDIDATES> refinement_wins = {};

#pragma omp parallel
    {
        std::array<unsigned long long, REFINEMENT_NCANDIDATES> local_refinement_wins = {};

#pragma omp for schedule(dynamic, 1)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = 0; i < width; ++i) {
                // хотим полного детерминизма, поэтому seed для рандома порождаем из номера итерации + из номера нашего пикселя,
                // тем самым получаем полный детерминизм и результат не зависит от числа ядер процессора и в теории может воспроизводиться даже на видеокарте
                FastRandom r(iter, j * width + i);

                // хотим попробовать улучшить текущие гипотезы рассмотрев взаимные комбинации следующих гипотез:
                float d0, dp, dr;
                vector3f n0, np, nr;

                {
                    // 1) текущей гипотезы (то что уже смогли найти)
                    d0 = depth_map.at<float>(j, i);
                    n0 = normal_map.at<vector3f>(j, i);

                    float depth_range = ref_depth_max - ref_depth_min;
                    float remaining_iterations_ratio = 1.0f;
                    if (NITERATIONS > 0) {
                        remaining_iterations_ratio = std::max(0.0f, std::min(1.0f - (float)iter / (float)NITERATIONS, 1.0f));
                    }
                    float depth_jitter = std::max(0.02f * depth_range, 0.5f * std::max(d0, ref_depth_min) * remaining_iterations_ratio);
                    float normal_jitter = std::max(0.05f, 0.5f * remaining_iterations_ratio);

                    // Без валидной прошлой гипотезы n0 останется нулевой нормалью из инициализации.
                    // Её нет смысла пертурбировать, поэтому пропускаем только достаточно ненулевые нормали,
                    // у которых norm2(n0) > 0.5f.
                    if (d0 != NO_DEPTH && norm2(n0) > 0.5f) {
                        // 2) случайной пертурбации текущей гипотезы (мутация и уточнение того что уже смогли найти)
                        dp = std::max(ref_depth_min, std::min(r.nextf(d0 - depth_jitter, d0 + depth_jitter), ref_depth_max));
                        np = cv::normalize(n0 + randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r) * normal_jitter);
                    } else {
                        dp = r.nextf(ref_depth_min, ref_depth_max);
                        np = randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r);
                    }

                    // 3) новой случайной гипотезы из фрустума поиска (новые идеи, вечный поиск во всем пространстве)
                    dr = r.nextf(ref_depth_min, ref_depth_max);
                    nr = randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r);
                }

                float best_depth = d0;
                vector3f best_normal = n0;
                float best_cost = cost_map.at<float>(j, i);
                size_t best_hi = 0;
                // Как будто нет смысла считать текущую гипотезу валидной, если нормаль вырождена.
                if (d0 == NO_DEPTH || best_cost == NO_COST || norm2(n0) < 0.5f) {
                    best_cost = NO_COST;
                    best_hi = std::numeric_limits<size_t>::max();
                }

                // В лоб refinement давал 3x3 = 9 комбинаций ({d0, dr, dp} x {n0, nr, np}), но добавив
                // статистику того, какая комбинация гипотез чаще всего побеждает, я заметил следующее...
                //
                // Полный лог теста со всеми 9-ью кандидатами saharov32 показал:
                // 1) на самом первом refinement (iter == 0) реально тащат качество только кандидаты,
                //    пробующие новые гипотезы: dp+np=25.97%, dr+np=25.72%, dp+nr=24.18%, dr+nr=24.13%;
                //    т.е. на старте мы действительно выигрываем от того, что перебираем всё декартово
                //    произведение из 9-ти комбинаций;
                // 2) уже на следующем refinement после первого propagation картинка резко меняется:
                //    d0+n0=58.10%, d0+np=25.06%, а dr+np, dp+nr, dr+nr падают примерно до 1%;
                // 3) на дальнейших итерациях всё ещё стабильно лучше всех себя показывают d0+n0 и d0+np,
                //    потом идёт dp+n0, а комбинации гипотез d0+nr и dr+n0 показывают себя в разы хуже.
                //
                // Из этого следует, что совсем выкидывать случайные комбинации гипотез нельзя, ибо они критичны
                // как раз для начального "бутстрапа" карты глубины, но и держать все 9 кандидатов на всех итерациях
                // невыгодно, потому что после первого refinement основная работа уже идет через локальное уточнение
                // текущей гипотезы.
                //
                // Поэтому я реализовал следующий компромисс:
                // 1) на iter == 0 мы оставляем все 9 кандидатов, чтобы старт был таким же "подробным",
                //    как в изначальной реализации;
                // 2) на всех следующих refinement переходим на более дешёвый по времени 5-кандидатный набор
                //    d0+n0, d0+np, dp+n0, dp+np, dr+nr (полный рандом был оставлен на всякий случай, 
                //    если текущая гипотеза оказалась прям уж совсем плохой).
                //
                // На тестах для saharov32 я получил следующие циферки:
                // - 5 кандидатов на всех итерациях: avg cost ~0.08906, good pixels ~76%, время работы теста ~190.9s;
                // - 9 кандидатов на всех итерациях: avg cost ~0.08445, good pixels ~78%, время работы теста ~247.1s;
                // - поэтапный "9 -> 5" режим:       avg cost ~0.08843, good pixels ~77%, время работы теста ~193.5s.
                //
                // Итого, видим, что выбранный мной компромисс работает достаточно быстрее перебора всех
                // 9-ти кандидатов. В качестве мы, конечно, проседаем относительно полного перебора,
                // но, насколько нам это критично, - это уже философский вопрос.
                const RefinementCandidateMode refinementCandidateMode = RefinementCandidateMode::StagedNineThenFive;

                const float cand_depths[REFINEMENT_NCANDIDATES] = { d0, d0, d0, dr, dr, dr, dp, dp, dp };
                const vector3f cand_normals[REFINEMENT_NCANDIDATES] = { n0, nr, np, n0, nr, np, n0, nr, np };
                const size_t cand_indices_all[] = {
                    REFINEMENT_KEEP_DEPTH_KEEP_NORMAL,
                    REFINEMENT_KEEP_DEPTH_RANDOM_NORMAL,
                    REFINEMENT_KEEP_DEPTH_PERTURBED_NORMAL,

                    REFINEMENT_RANDOM_DEPTH_KEEP_NORMAL,
                    REFINEMENT_RANDOM_DEPTH_RANDOM_NORMAL,
                    REFINEMENT_RANDOM_DEPTH_PERTURBED_NORMAL,

                    REFINEMENT_PERTURBED_DEPTH_KEEP_NORMAL,
                    REFINEMENT_PERTURBED_DEPTH_RANDOM_NORMAL,
                    REFINEMENT_PERTURBED_DEPTH_PERTURBED_NORMAL
                };
                const size_t cand_indices_after_first[] = {
                    REFINEMENT_KEEP_DEPTH_KEEP_NORMAL,
                    REFINEMENT_KEEP_DEPTH_PERTURBED_NORMAL,

                    REFINEMENT_PERTURBED_DEPTH_KEEP_NORMAL,
                    REFINEMENT_PERTURBED_DEPTH_PERTURBED_NORMAL,

                    REFINEMENT_RANDOM_DEPTH_RANDOM_NORMAL
                };
                const size_t* cand_indices = nullptr;
                size_t cand_count = 0;
                switch (refinementCandidateMode) {
                    case RefinementCandidateMode::FiveCandidatesOnly:
                        cand_indices = cand_indices_after_first;
                        cand_count = sizeof(cand_indices_after_first) / sizeof(cand_indices_after_first[0]);
                        break;
                    case RefinementCandidateMode::NineCandidatesOnly:
                        cand_indices = cand_indices_all;
                        cand_count = sizeof(cand_indices_all) / sizeof(cand_indices_all[0]);
                        break;
                    case RefinementCandidateMode::StagedNineThenFive:
                        cand_indices = cand_indices_after_first;
                        cand_count = sizeof(cand_indices_after_first) / sizeof(cand_indices_after_first[0]);
                        if (iter == 0) {
                            cand_indices = cand_indices_all;
                            cand_count = sizeof(cand_indices_all) / sizeof(cand_indices_all[0]);
                        }
                        break;
                }

                for (size_t ci = 0; ci < cand_count; ++ci) {
                    size_t hi = cand_indices[ci];
                    // эту комбинацию-гипотезу мы сейчас рассматриваем как очередного кандидата
                    float d = cand_depths[hi];
                    vector3f n = cand_normals[hi];

                    // Все ещё нет смысла работать с почти нулевыми нормалями.
                    if (d == NO_DEPTH || norm2(n) < 0.5f)
                        continue;

                    // оцениваем cost для каждого соседа
                    std::vector<float> costs;
                    for (size_t ni = 0; ni < ncameras; ++ni) {
                        if (ni == ref_cam)
                            continue;

                        float costi = estimateCost(i, j, d, n, ni);
                        if (costi == NO_COST)
                            continue;

                        costs.push_back(costi);
                    }

                    // объединяем cost-ы всех соседей в одну общую оценку качества текущей гипотезы (условно "усредняем")
                    float total_cost = avgCost(costs);

                    // WTA (winner takes all)
                    if (total_cost < best_cost) {
                        best_depth = d;
                        best_normal = n;
                        best_cost = total_cost; // добавьте подсчет статистики, какая комбинация гипотез чаще всего побеждает? есть ли комбинации на которых мы можем сэкономить? а какие гипотезы при refinement рассматривает например
                                                // Colmap?
                        // Судя по https://github.com/colmap/colmap/blob/main/src/colmap/mvs/patch_match_cuda.cu#L1083-L1092,
                        // Colmap использует:
                        // - 4 гипотезы, аналогичные нашим {d0, dp} x {n0, np};
                        // - 1 гипотезу, полученную от предыдущего пикселя:
                        //   глубина получается пересечением луча текущего пикселя с плоскостью предыдущего пикселя,
                        //   а нормаль берём просто от предыдущего пикселя.
                        best_hi = hi;
                    }
                }

                depth_map.at<float>(j, i) = best_depth;
                normal_map.at<vector3f>(j, i) = best_normal;
                cost_map.at<float>(j, i) = best_cost;

                if (best_hi != std::numeric_limits<size_t>::max()) {
                    local_refinement_wins[best_hi] += 1;
                }
            }
        }

#pragma omp critical
        {
            for (size_t hi = 0; hi < refinement_wins.size(); ++hi) {
                refinement_wins[hi] += local_refinement_wins[hi];
            }
        }
    }

    verbose_cout << "refinement done in " << t.elapsed() << " s: ";
#ifdef VERBOSE_LOGGING
    printCurrentStats();
    const char* hnames[REFINEMENT_NCANDIDATES] = { "d0+n0", "d0+nr", "d0+np", "dr+n0", "dr+nr", "dr+np", "dp+n0", "dp+nr", "dp+np" };
    unsigned long long wins_total = 0;
    for (size_t hi = 0; hi < refinement_wins.size(); ++hi) {
        wins_total += refinement_wins[hi];
    }
    if (wins_total > 0) {
        std::vector<std::pair<unsigned long long, size_t>> sorted;
        sorted.reserve(refinement_wins.size());
        for (size_t hi = 0; hi < refinement_wins.size(); ++hi) {
            sorted.push_back(std::make_pair(refinement_wins[hi], hi));
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

        verbose_cout << "    refinement winners:";
        for (size_t shi = 0; shi < std::min<size_t>(3, sorted.size()); ++shi) {
            if (sorted[shi].first == 0)
                break;
            double ratio = sorted[shi].first * 100.0 / wins_total;
            verbose_cout << " " << hnames[sorted[shi].second] << "=" << sorted[shi].first << " (" << ratio << "%)";
        }
        verbose_cout << std::endl;
    }
#endif
#ifdef DEBUG_DIR
    debugCurrentPoints(to_string(ref_cam) + "_" + to_string(iter) + "_refinement");
#endif
}

void PMDepthMapsBuilder::propagation()
{
    timer t;
    verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": propagation..." << std::endl;

    for (int chessboard_pattern_step = 0; chessboard_pattern_step < 2; ++chessboard_pattern_step) {
        const cv::Mat propagation_depth_map = depth_map.clone();
        const cv::Mat propagation_normal_map = normal_map.clone();
        const cv::Mat propagation_cost_map = cost_map.clone();

#pragma omp parallel for schedule(dynamic, 1)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = (j + chessboard_pattern_step) % 2; i < width; i += 2) {
                enum
                {
                    ACMH_UP_NEAR,
                    ACMH_UP_FAR,
                    ACMH_DOWN_NEAR,
                    ACMH_DOWN_FAR,
                    ACMH_LEFT_NEAR,
                    ACMH_LEFT_FAR,
                    ACMH_RIGHT_NEAR,
                    ACMH_RIGHT_FAR,
                    ACMH_NCANDIDATES
                };

                float hypos_depth[ACMH_NCANDIDATES] = {};
                vector3f hypos_normal[ACMH_NCANDIDATES];
                float hypos_donor_cost[ACMH_NCANDIDATES] = {};
                bool hypos_valid[ACMH_NCANDIDATES] = {};

                auto tryUpdateBestDonor = [&](size_t hypo_idx, ptrdiff_t ni, ptrdiff_t nj) {
                    if (ni < 0 || ni >= width || nj < 0 || nj >= height)
                        return;

                    float d = propagation_depth_map.at<float>(nj, ni);
                    if (d == NO_DEPTH)
                        return;

                    vector3f n = propagation_normal_map.at<vector3f>(nj, ni);
                    float donor_cost = propagation_cost_map.at<float>(nj, ni);
                    if (donor_cost == NO_COST || norm2(n) < 0.5f)
                        return;

                    if (!hypos_valid[hypo_idx] || donor_cost < hypos_donor_cost[hypo_idx]) {
                        hypos_valid[hypo_idx] = true;
                        hypos_depth[hypo_idx] = d;
                        hypos_normal[hypo_idx] = n;
                        hypos_donor_cost[hypo_idx] = donor_cost;
                    }
                };

                if (j > 2) {
                    for (int k = 0; k < 11; ++k) {
                        tryUpdateBestDonor(ACMH_UP_FAR, i, j - (3 + 2 * k));
                    }
                }
                if (j < height - 3) {
                    for (int k = 0; k < 11; ++k) {
                        tryUpdateBestDonor(ACMH_DOWN_FAR, i, j + (3 + 2 * k));
                    }
                }
                if (i > 2) {
                    for (int k = 0; k < 11; ++k) {
                        tryUpdateBestDonor(ACMH_LEFT_FAR, i - (3 + 2 * k), j);
                    }
                }
                if (i < width - 3) {
                    for (int k = 0; k < 11; ++k) {
                        tryUpdateBestDonor(ACMH_RIGHT_FAR, i + (3 + 2 * k), j);
                    }
                }

                if (j > 0) {
                    tryUpdateBestDonor(ACMH_UP_NEAR, i, j - 1);
                    for (int k = 1; k <= 3; ++k) {
                        tryUpdateBestDonor(ACMH_UP_NEAR, i - k, j - (1 + k));
                        tryUpdateBestDonor(ACMH_UP_NEAR, i + k, j - (1 + k));
                    }
                }
                if (j < height - 1) {
                    tryUpdateBestDonor(ACMH_DOWN_NEAR, i, j + 1);
                    for (int k = 1; k <= 3; ++k) {
                        tryUpdateBestDonor(ACMH_DOWN_NEAR, i - k, j + (1 + k));
                        tryUpdateBestDonor(ACMH_DOWN_NEAR, i + k, j + (1 + k));
                    }
                }
                if (i > 0) {
                    tryUpdateBestDonor(ACMH_LEFT_NEAR, i - 1, j);
                    for (int k = 1; k <= 3; ++k) {
                        tryUpdateBestDonor(ACMH_LEFT_NEAR, i - (1 + k), j - k);
                        tryUpdateBestDonor(ACMH_LEFT_NEAR, i - (1 + k), j + k);
                    }
                }
                if (i < width - 1) {
                    tryUpdateBestDonor(ACMH_RIGHT_NEAR, i + 1, j);
                    for (int k = 1; k <= 3; ++k) {
                        tryUpdateBestDonor(ACMH_RIGHT_NEAR, i + (1 + k), j - k);
                        tryUpdateBestDonor(ACMH_RIGHT_NEAR, i + (1 + k), j + k);
                    }
                }

                // TODO 301 - сделайте вместо наивного переноса depth+normal в наш пиксель - логику про "пересекли луч из нашего пикселя с плоскостью которую задает донор-сосед" и оценку cost в нашей точке тогда можно провести для более
                // релевантной точки-пересечения

                float best_depth = depth_map.at<float>(j, i);
                vector3f best_normal = normal_map.at<vector3f>(j, i);
                float best_cost = cost_map.at<float>(j, i);
                if (best_depth == NO_DEPTH) {
                    best_cost = NO_COST;
                }

                for (size_t hi = 0; hi < ACMH_NCANDIDATES; ++hi) {
                    if (!hypos_valid[hi])
                        continue;

                    // эту гипотезу мы сейчас рассматриваем как очередного кандидата
                    float d = hypos_depth[hi];
                    vector3f n = hypos_normal[hi];

                    // оцениваем cost для каждого соседа
                    std::vector<float> costs;
                    for (size_t ni = 0; ni < ncameras; ++ni) {
                        if (ni == ref_cam)
                            continue;

                        float costi = estimateCost(i, j, d, n, ni);
                        if (costi == NO_COST)
                            continue;

                        costs.push_back(costi);
                    }

                    // объединяем cost-ы всех соседей в одну общую оценку качества текущей гипотезы (условно "усредняем")
                    float total_cost = avgCost(costs);

                    // WTA (winner takes all)
                    if (total_cost < best_cost) {
                        best_depth = d;
                        best_normal = n;
                        best_cost = total_cost;
                    }
                }

                depth_map.at<float>(j, i) = best_depth;
                normal_map.at<vector3f>(j, i) = best_normal;
                cost_map.at<float>(j, i) = best_cost;
            }
        }
    }

    verbose_cout << "propagation done in " << t.elapsed() << " s: ";
#ifdef VERBOSE_LOGGING
    printCurrentStats();
#endif
#ifdef DEBUG_DIR
    debugCurrentPoints(to_string(ref_cam) + "_" + to_string(iter) + "_propagation");
#endif
}

bool sampleGreyBilinear(const cv::Mat& img, double x, double y, float& intensity)
{
    rassert(img.type() == CV_8UC1, 9132784623824);

    if (x < 0.5 || x > img.cols - 0.5 || y < 0.5 || y > img.rows - 0.5) {
        return false;
    }

    double gx = x - 0.5;
    double gy = y - 0.5;

    ptrdiff_t x0 = std::floor(gx);
    ptrdiff_t y0 = std::floor(gy);
    if (x0 < 0 || x0 >= img.cols || y0 < 0 || y0 >= img.rows) {
        return false;
    }

    ptrdiff_t x1 = std::min<ptrdiff_t>(x0 + 1, img.cols - 1);
    ptrdiff_t y1 = std::min<ptrdiff_t>(y0 + 1, img.rows - 1);

    float tx = gx - x0;
    float ty = gy - y0;

    float i00 = img.at<unsigned char>(y0, x0) / 255.0f;
    float i10 = img.at<unsigned char>(y0, x1) / 255.0f;
    float i01 = img.at<unsigned char>(y1, x0) / 255.0f;
    float i11 = img.at<unsigned char>(y1, x1) / 255.0f;

    float top = i00 * (1.0f - tx) + i10 * tx;
    float bottom = i01 * (1.0f - tx) + i11 * tx;
    intensity = top * (1.0f - ty) + bottom * ty;
    return true;
}

float PMDepthMapsBuilder::estimateCost(ptrdiff_t i, ptrdiff_t j, double d, const vector3d& global_normal, size_t neighb_cam)
{
    vector3d pixel(i + 0.5, j + 0.5, d);
    vector3d global_point = unproject(pixel, calibration, cameras_PtoWorld[ref_cam]);

    if (!(i - COST_PATCH_RADIUS >= 0 && i + COST_PATCH_RADIUS < width))
        return NO_COST;
    if (!(j - COST_PATCH_RADIUS >= 0 && j + COST_PATCH_RADIUS < height))
        return NO_COST;

    std::vector<float> patch0, patch1;

    for (ptrdiff_t dj = -COST_PATCH_RADIUS; dj <= COST_PATCH_RADIUS; ++dj) {
        for (ptrdiff_t di = -COST_PATCH_RADIUS; di <= COST_PATCH_RADIUS; ++di) {
            ptrdiff_t ni = i + di;
            ptrdiff_t nj = j + dj;

            patch0.push_back(cameras_imgs_grey[ref_cam].at<unsigned char>(nj, ni) / 255.0f);

            vector3d point_on_ray = unproject(vector3d(ni + 0.5, nj + 0.5, 1.0), calibration, cameras_PtoWorld[ref_cam]);
            vector3d camera_center = cameras_O[ref_cam];

            vector3d ray_dir = cv::normalize(point_on_ray - camera_center);
            vector3d ray_org = camera_center;

            vector3d global_intersection;
            if (!intersectWithPlane(global_point, global_normal, ray_org, ray_dir, global_intersection))
                return NO_COST; // луч не пересек плоскость (например наблюдаем ее под близким к прямому углу)

            rassert(neighb_cam != ref_cam, 2334195412410286);
            vector3d neighb_proj = project(global_intersection, calibration, cameras_PtoLocal[neighb_cam]);
            if (neighb_proj[2] < 0.0)
                return NO_COST;

            double x = neighb_proj[0];
            double y = neighb_proj[1];

            float intensity = 0.0f;
            if (!sampleGreyBilinear(cameras_imgs_grey[neighb_cam], x, y, intensity))
                return NO_COST;
            patch1.push_back(intensity);
        }
    }

    rassert(patch0.size() == patch1.size(), 12489185129326);
    size_t n = patch0.size();
    float mean0 = 0.0f;
    float mean1 = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        float a = patch0[k];
        float b = patch1[k];
        mean0 += a;
        mean1 += b;
    }
    mean0 /= n;
    mean1 /= n;

    float covar = 0.0f;
    float var0 = 0.0f;
    float var1 = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        float da = patch0[k] - mean0;
        float db = patch1[k] - mean1;
        covar += da * db;
        var0 += da * da;
        var1 += db * db;
    }

    float denom = std::sqrt(var0 * var1);
    if (denom < 1e-12f) {
        return NO_COST;
    }
    float zncc = covar / denom;

    // ZNCC в диапазоне [-1; 1], 1: идеальное совпадение, -1: ничего общего
    rassert(zncc == zncc, 23141241210380); // проверяем что не nan
    zncc = std::max(-1.0f, std::min(1.0f, zncc));
    rassert(zncc >= -1.0 && zncc <= 1.0, 141251251541357);

    // переводим в cost от [0; 1] (NO_COST=1)
    // чем ближе cost к нулю - тем лучше сопоставление
    float cost = (1.0f - zncc) / 2.0f;
    rassert(cost >= 0.0f && cost <= NO_COST, 23123912049102361);

    return cost;
}

float PMDepthMapsBuilder::avgCost(std::vector<float>& costs)
{
    if (costs.size() == 0)
        return NO_COST;

    std::sort(costs.begin(), costs.end());

    float best_cost = costs[0];

    float cost_sum = best_cost;
    float cost_w = 1.0f;

    float cost_limit = std::min(COSTS_ABS_LIMIT, best_cost * COSTS_K_RATIO);
    for (size_t ci = 1; ci < costs.size() && ci < COSTS_BEST_K_LIMIT; ++ci) {
        float cost = costs[ci];
        if (cost > cost_limit)
            continue;
        cost_sum += cost;
        cost_w += 1.0f;
    }

    float avg_cost = cost_sum / cost_w;

    // а что если добавить какой-нибудь бонус в случае если больше чем Х камер засчиталось? улучшается/ухудшается ли от этого что-то на herzjezu25? а при большем числе фотографий

    // Вообще говоря, "много камер засчиталось" и "каждая из этих камер дала действительно хороший cost" - не одно и то же.
    // Поэтому бонус не следует делать слишком большим, ибо иначе этот бонус бы награждать кол-во камер само по себе
    // и поощрять гипотезы, у которых совпадение посредственное, но их подтвердило много камер.
    // Итого, чтобы этот бонус не оказал нам медвежью услугу, предлагатся:
    // 1) применять бонус, если число учтённых камер не меньше COST_SUPPORT_BONUS_MIN_SUPPORTING_CAMERAS;
    // 2) зажать бонус сверху значением COST_SUPPORT_BONUS_MAX, чтобы он не превращался в новый доминирующий критерий.
    //
    // На тестах я получил следующие циферки:
    // - saharov32, 5 камер:
    //   bonus off: avg cost ~0.08843, good pixels ~77%;
    //   bonus on:  avg cost ~0.08842, good pixels ~77%;
    // - saharov32, 9 камер:
    //   bonus off: avg cost ~0.07803, good pixels ~80%;
    //   bonus on:  avg cost ~0.07798, good pixels ~80%;
    // - herzjesu25, 5 камер:
    //   bonus off: avg cost ~0.02746, good pixels ~93%;
    //   bonus on:  avg cost ~0.02732, good pixels ~93%;
    // - herzjesu25, 9 камер:
    //   bonus off: avg cost ~0.02768, good pixels ~93%;
    //   bonus on:  avg cost ~0.02733, good pixels ~93%.
    //
    // Итого, от этого бонуса виден один лишь шум :) никаких существенных ни ухудшений, ни улучшений.

#if COST_SUPPORT_BONUS_ENABLED
    if (cost_w >= COST_SUPPORT_BONUS_MIN_SUPPORTING_CAMERAS) {
        float supporting_cameras = cost_w - COST_SUPPORT_BONUS_MIN_SUPPORTING_CAMERAS + 1.0f;
        float support_bonus = std::min(COST_SUPPORT_BONUS_MAX, supporting_cameras * COST_SUPPORT_BONUS_PER_EXTRA_CAMERA);
        avg_cost = std::max(0.0f, avg_cost - support_bonus);
    }
#endif

    return avg_cost;
}

void PMDepthMapsBuilder::printCurrentStats()
{
    double costs_sum = 0.0;
    double costs_n = 0.0;
    double good_costs_sum = 0.0;
    double good_costs_n = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : costs_sum, costs_n, good_costs_sum, good_costs_n)
    for (ptrdiff_t j = 0; j < height; ++j) {
        for (ptrdiff_t i = 0; i < width; ++i) {
            float d = depth_map.at<float>(j, i);
            if (d == NO_DEPTH)
                continue;

            float cost = cost_map.at<float>(j, i);
            if (cost == NO_COST)
                continue;

            costs_sum += cost;
            costs_n += 1.0;

            if (cost < GOOD_COST) {
                good_costs_sum += cost;
                good_costs_n += 1.0;
            }
        }
    }
    double ntotal = width * height;
    verbose_cout << to_percent(costs_n, ntotal) << "% pixels with " << (costs_sum / costs_n) << " avg cost, ";
    verbose_cout << to_percent(good_costs_n, ntotal) << "% pixels with good " << (good_costs_sum / good_costs_n) << " avg cost";
    verbose_cout << std::endl;
}

void PMDepthMapsBuilder::debugCurrentPoints(const std::string& label)
{
    std::vector<cv::Vec3d> point_cloud_all;
    std::vector<cv::Vec3b> point_cloud_all_bgr;
    std::vector<cv::Vec3d> point_cloud_all_normal;

    std::vector<cv::Vec3d> point_cloud_good;
    std::vector<cv::Vec3b> point_cloud_good_bgr;
    std::vector<cv::Vec3b> point_cloud_good_cost;
    std::vector<cv::Vec3d> point_cloud_good_normal;

    for (ptrdiff_t j = 0; j < height; ++j) {
        for (ptrdiff_t i = 0; i < width; ++i) {
            float depth = depth_map.at<float>(j, i);
            float cost = cost_map.at<float>(j, i);
            vector3d normal = normal_map.at<vector3f>(j, i);

            if (depth == NO_DEPTH || cost == NO_COST)
                continue;

            cv::Vec3d p = unproject(vector3d(i + 0.5, j + 0.5, depth), calibration, cameras_PtoWorld[ref_cam]);
            cv::Vec3b bgr = cameras_imgs[ref_cam].at<cv::Vec3b>(j, i);
            point_cloud_all.push_back(p);
            point_cloud_all_bgr.push_back(bgr);
            point_cloud_all_normal.push_back(normal);

            if (cost > GOOD_COST)
                continue;

            cv::Vec3b cost_bgr;
            for (int c = 0; c < 3; ++c) {
                cost_bgr[c] = (unsigned char)(255.0f * (1.0f - cost / GOOD_COST));
            }
            point_cloud_good.push_back(p);
            point_cloud_good_bgr.push_back(bgr);
            point_cloud_good_cost.push_back(cost_bgr);
            point_cloud_good_normal.push_back(normal);
        }
    }

    exportPointCloud(point_cloud_all, DEBUG_DIR + label + "_all_rgb.ply", point_cloud_all_bgr, point_cloud_all_normal);
    exportPointCloud(point_cloud_good, DEBUG_DIR + label + "_good_rgb.ply", point_cloud_good_bgr, point_cloud_good_normal);
    exportPointCloud(point_cloud_good, DEBUG_DIR + label + "_good_costs.ply", point_cloud_good_cost, point_cloud_good_normal);
}

void PMDepthMapsBuilder::buildGoodPoints(const cv::Mat& depth_map, const cv::Mat& normal_map, const cv::Mat& cost_map, const cv::Mat& img, const phg::Calibration& calibration, const matrix34d& PtoWorld, std::vector<cv::Vec3d>& points,
    std::vector<cv::Vec3b>& colors, std::vector<cv::Vec3d>& normals)
{
    ptrdiff_t width = calibration.width();
    ptrdiff_t height = calibration.height();

    for (ptrdiff_t j = 0; j < height; ++j) {
        for (ptrdiff_t i = 0; i < width; ++i) {
            float depth = depth_map.at<float>(j, i);
            float cost = cost_map.at<float>(j, i);
            vector3d normal = normal_map.at<vector3f>(j, i);

            if (depth == NO_DEPTH || cost == NO_COST)
                continue;

            cv::Vec3d p = unproject(vector3d(i + 0.5, j + 0.5, depth), calibration, PtoWorld);
            cv::Vec3b bgr = img.at<cv::Vec3b>(j, i);

            if (cost > GOOD_COST)
                continue;

            points.push_back(p);
            colors.push_back(bgr);
            normals.push_back(normal);
        }
    }
}

}
