#include "pm_depth_maps.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <libutils/fast_random.h>
#include <libutils/timer.h>
#include <opencv2/core.hpp>
#include <phg/utils/point_cloud_export.h>
#include <utility>

#include "phg/sfm/defines.h"
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

// 101 реализуйте unproject (вам поможет тест на идемпотентность project -> unproject в test_depth_maps_pm)
vector3d unproject(const vector3d& pixel, const phg::Calibration& calibration, const matrix34d& PtoWorld)
{
    double depth = pixel[2]; // на самом деле это не глубина, это координата по оси +Z (вдоль которой смотрит камера в ее локальной системе координат)
;
    vector3d local_point = calibration.unproject(vector2d(pixel[0], pixel[1])); // 102 пустите луч pixel из calibration а затем возьмите ан нем точку у которой по оси +Z координата=depth
    local_point *= depth / local_point[2];

    vector3d global_point = PtoWorld * vector4d(local_point[0], local_point[1], local_point[2], 1.0); // 103 переведите точку из локальной системы в глобальную

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

#pragma omp parallel for schedule(dynamic, 1)
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

                // 2) случайной пертурбации текущей гипотезы (мутация и уточнение того что уже смогли найти)
                float mul = std::exp(-1.0f * iter / NITERATIONS);
                dp = r.nextf(d0 * (1.0f - 0.5f * mul), d0 * (1.0f + 0.5f * mul)); // 104: сделайте так чтобы отклонение было тем меньше, чем номер итерации ближе к NITERATIONS, улучшило ли это результат?
                np = cv::normalize(n0 + randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r) * 0.5 * mul); // 105: сделайте так чтобы отклонение было тем меньше, чем номер итерации ближе к NITERATIONS, улучшило ли это результат?
                // Оба взвешивания улучшают конечную good avg cost на 0.3-0.4% (0.0685 -> 0.0682 -> 0.068)
                dp = std::max(ref_depth_min, std::min(ref_depth_max, dp));

                // 3) новой случайной гипотезы из фрустума поиска (новые идеи, вечный поиск во всем пространстве)
                // 106: создайте случайную гипотезу dr+nr, вам поможет:
                //  - r.nextf(...)
                //  - ref_depth_min, ref_depth_max
                //  - randomNormalObservedFromCamera - поможет создать нормаль которая гарантированно смотрит на нас
                dr = r.nextf(ref_depth_min, ref_depth_max);
                nr = randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r);
            }

            float best_depth = d0;
            vector3f best_normal = n0;
            float best_cost = cost_map.at<float>(j, i);
            size_t best_idx = 0;
            if (d0 == NO_DEPTH) {
                best_cost = NO_COST;
            }

            constexpr size_t n_hypot = 6;
            const std::array<std::pair<float, vector3f>, n_hypot> hypots = {
                std::pair<float, vector3f>{d0, n0},
                // {d0, nr},
                {d0, np},
                // {dr, n0},
                {dr, nr},
                {dr, np},
                // {dp, n0},
                {dp, nr},
                {dp, np},
            };

            // перебираем все комбинации этих гипотез, т.е. 3х3=9 вариантов
            for (size_t hi = 0; hi < n_hypot; ++hi) {
                // эту комбинацию-гипотезу мы сейчас рассматриваем как очередного кандидата
                auto [d, n] = hypots[hi];

                // оцениваем cost для каждого соседа
                std::vector<float> costs;
                for (size_t ni = 0; ni < ncameras; ++ni) {
                    if (ni == ref_cam)
                        continue;

                    float costi = estimateCost(i, j, d, n, ni);
                    costs.push_back(costi);
                }

                // объединяем cost-ы всех соседей в одну общую оценку качества текущей гипотезы (условно "усредняем")
                float total_cost = avgCost(costs);

                // WTA (winner takes all)
                if (total_cost < best_cost) {
                    best_depth = d;
                    best_normal = n;
                    best_cost = total_cost; // 206: добавьте подсчет статистики, какая комбинация гипотез чаще всего побеждает? есть ли комбинации на которых мы можем сэкономить? а какие гипотезы при refinement рассматривает например
                                            // Colmap?

                    // Получается примерно такое распределение:
                    // x   n0    nr    np
                    // d0  0.7   0.02  0.1
                    // dr  0.01  0.05  0.05
                    // dp  0.01  0.03  0.03
                    // Если что-то отбрасывать, то первые на очереди n0 + dr/dp, затем nr + d0. Остальные я бы оставил

                    // В colmap я накопал вот эту строчку: https://github.com/colmap/colmap/blob/main/src/colmap/mvs/patch_match_cuda.cu#L1083
                    // После просмотра кода по диагонали у меня сложилось впечатление, что у них:
                    // prev --- наш 0
                    // curr --- наш p
                    // rand --- наш r
                    // То есть они смотрят вот такие пары:
                    // x   n0  nr  np
                    // d0  +   -   -
                    // dr  -   +   +
                    // dp  -   +   +
                    // От нашей версии это отличается отсутствием пары np+d0

                    best_idx = hi;
                }
            }

            depth_map.at<float>(j, i) = best_depth;
            normal_map.at<vector3f>(j, i) = best_normal;
            cost_map.at<float>(j, i) = best_cost;
            winner_hyp_hits[best_idx] += 1;
        }
    }

    total_hyps += width * height;

    verbose_cout << "refinement done in " << t.elapsed() << " s:\n";
#ifdef VERBOSE_LOGGING
    printCurrentStats();
#endif
#ifdef DEBUG_DIR
    debugCurrentPoints(to_string(ref_cam) + "_" + to_string(iter) + "_refinement");
#endif
}

bool PMDepthMapsBuilder::tryToPropagateDonor(ptrdiff_t ni, ptrdiff_t nj, int chessboard_pattern_step, std::vector<float>& hypos_depth, std::vector<vector3f>& hypos_normal, std::vector<float>& hypos_cost, std::vector<vector2d> &hypos_pos)
{
    // rassert-ы или любой другой способ явной фиксации инвариантов со встроенной их проверкой в runtime -
    // это очень приятный способ ускорить отладку и гарантировать что ожидания в голове сойдутся с реальностью в коде,
    // а если разойдутся - то узнать об этом в самом первом сломавшемся предположении
    // (в данном случае мы явно проверяем что нигде не промахнулись и все соседи - другого шахматного цвета)
    // пусть лучше эта проверка упадет, мы сразу это заметим и отладим, чем бага будет тихо портить результаты
    // а мы это может быть даже не заметим
    rassert((ni + nj) % 2 != chessboard_pattern_step, 2391249129510120);

    if (ni < 0 || ni >= width || nj < 0 || nj >= height)
        return false;

    float d = depth_map.at<float>(nj, ni);
    if (d == NO_DEPTH)
        return false;

    vector3f n = normal_map.at<vector3f>(nj, ni);
    float cost = cost_map.at<float>(nj, ni);

    hypos_depth.push_back(d);
    hypos_normal.push_back(n);
    hypos_cost.push_back(cost);   
    hypos_pos.emplace_back(ni, nj);
    return true;
}

bool PMDepthMapsBuilder::tryToPropagateDonorGroup(
    ptrdiff_t x, ptrdiff_t y, int chessboard_pattern_step, std::vector<float>& hypos_depth, std::vector<vector3f>& hypos_normal, std::vector<float>& hypos_cost, std::vector<vector2d> &hypos_pos, const std::initializer_list<std::pair<ptrdiff_t, ptrdiff_t>>& offsets)
{

    int new_idx = hypos_cost.size();
    float min_cost = NO_COST;
    int best_idx = -1;

    for (auto [dx, dy] : offsets) {
        if (!tryToPropagateDonor(x + dx, y + dy, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos)) continue;
        if (hypos_cost.back() < min_cost) {
            min_cost = hypos_cost.back();
            best_idx = hypos_cost.size() - 1;
        }
    }
    if (hypos_cost.size() == new_idx) return false;

    auto retain_best = [best_idx, new_idx](auto &vec) {
        std::swap(vec[new_idx], vec[best_idx]);
        vec.resize(new_idx + 1);
    };
    retain_best(hypos_depth);
    retain_best(hypos_normal);
    retain_best(hypos_cost);
    retain_best(hypos_pos);

    return true;
}

void PMDepthMapsBuilder::propagation()
{
    timer t;
    verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": propagation..." << std::endl;

    for (int chessboard_pattern_step = 0; chessboard_pattern_step < 2; ++chessboard_pattern_step) {
#pragma omp parallel for schedule(dynamic, 1)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = (j + chessboard_pattern_step) % 2; i < width; i += 2) {
                std::vector<float> hypos_depth;
                std::vector<vector3f> hypos_normal;
                std::vector<float> hypos_cost;
                std::vector<vector2d> hypos_pos;

                /* 4 прямых соседа A, 8 соседей B через диагональ, 4 соседа C вдалеке (условный рисунок для PROPAGATION_STEP=5):
                 * (удобно подсвечивать через Ctrl+F)
                 *         center
                 *           |
                 *           v
                 * o o o o o C o o o o o
                 * o o o o o o o o o o o
                 * o o o o o o o v o o o
                 * o o o o B o B o v o o
                 * o o o B o A o B o o o
                 * C o o o A . A o o o C  <- center
                 * o o o B o A o B o o o
                 * o o o o B o B o v o o
                 * o o o o o o o v o o o
                 * o o o o o o o o o o o
                 * o o o o o C o o o o o
                 */
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{0, 1}, {-1, 2}, {1, 2}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{1, 0}, {2, -1}, {2, 1}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{0, -1}, {-1, -2}, {1, -2}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{-1, 0}, {-2, -1}, {-2, 1}});
                
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{0, PROPAGATION_STEP}, {0, PROPAGATION_STEP + 2}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{0, -PROPAGATION_STEP}, {0, -PROPAGATION_STEP - 2}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{PROPAGATION_STEP, 0}, {PROPAGATION_STEP + 2, 0}});
                tryToPropagateDonorGroup(i, j, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos, {{-PROPAGATION_STEP, 0}, {-PROPAGATION_STEP - 2, 0}});

                // tryToPropagateDonor(i - 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 0, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 0, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);

                // tryToPropagateDonor(i - 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i - 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i - 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i - 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);

                // // в таких случаях очень приятно использовать множественный курсор (чтобы скопировав четыре строки выше, затем просто колесиком мышки сделать четыре каретки для того чтобы дважды вставить *PROPAGATION_STEP):
                // tryToPropagateDonor(i - 1 * PROPAGATION_STEP, j + 0 * PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 0 * PROPAGATION_STEP, j - 1 * PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 1 * PROPAGATION_STEP, j + 0 * PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);
                // tryToPropagateDonor(i + 0 * PROPAGATION_STEP, j + 1 * PROPAGATION_STEP, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost, hypos_pos);

                // 201 переделайте чтобы было как в ACMH:
                // 202 - паттерн донорства
                // 203 - логика про "берем 8 лучших по их личной оценке - по их личному cost" и только их примеряем уже на себя для рассчета cost в нашей точке
                // 301 - сделайте вместо наивного переноса depth+normal в наш пиксель - логику про "пересекли луч из нашего пикселя с плоскостью которую задает донор-сосед" и оценку cost в нашей точке тогда можно провести для более
                // релевантной точки-пересечения

                float best_depth = depth_map.at<float>(j, i);
                vector3f best_normal = normal_map.at<vector3f>(j, i);
                float best_cost = cost_map.at<float>(j, i);
                if (best_depth == NO_DEPTH) {
                    best_cost = NO_COST;
                }

                vector3d center = camera_centers[ref_cam];
                vector3d target1 = unproject(vector3d{i + 0.5, j + 0.5, 1.0}, calibration, cameras_PtoWorld[ref_cam]);
                vector3d target_dir = cv::normalize(target1 - center);

                for (size_t hi = 0; hi < hypos_depth.size(); ++hi) {
                    // эту гипотезу мы сейчас рассматриваем как очередного кандидата
                    float d = hypos_depth[hi];
                    vector3f n = hypos_normal[hi];

                    double c = n.dot(target_dir);
                    if (c > 1e-4) {
                        vector2d hpx = hypos_pos[hi];
                        vector3d hpos = unproject(vector3d{hpx[0] + 0.5, hpx[1] + 0.5, d}, calibration, cameras_PtoWorld[ref_cam]);
                        d = n.dot(hpos - center) / c; 
                    }

                    // cv::inter
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
            vector3d camera_center = camera_centers[ref_cam]; // 204: это немного неестественный способ, можно поправить его на более явный вариант, например хранить центр камер в поле cameras_O

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

            // 205: замените этот наивный вариант nearest neighbor сэмплирования текстуры на билинейную интерполяцию (учтите что центр пикселя - .5 после запятой)            
            // Смещение на 2 для корректного округления около нуля
            ptrdiff_t bx = (ptrdiff_t)(x + 1.5) - 2;
            float alpha = x - bx - 0.5;
            ptrdiff_t by = (ptrdiff_t)(y + 1.5) - 2;
            float beta = y - by - 0.5;

            auto read_texture = [&tex = cameras_imgs_grey[neighb_cam]](ptrdiff_t u, ptrdiff_t v) {
                if (v < 0 || v >= tex.rows || u < 0 || u >= tex.cols)
                    return NO_COST;
                else 
                    return tex.at<unsigned char>(v, u) / 255.0f;
            };

            patch1.push_back(
                (read_texture(bx, by) * (1 - alpha) + read_texture(bx + 1, by) * alpha) * (1 - beta) + 
                (read_texture(bx, by + 1) * (1 - alpha) + read_texture(bx + 1, by + 1) * alpha) * beta
            );
        }
    }

    // 109: реализуйте ZNCC https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    // или слайд #25 в лекции 5 про SGM и Cost-функции - https://my.compscicenter.ru/attachments/classes/slides_w2n8WNLY/photogrammetry_lecture_090321.pdf
    rassert(patch0.size() == patch1.size(), 12489185129326);
    size_t n = patch0.size();
    float mean0 = 0.0f;
    float mean1 = 0.0f;
    // ...
    for (size_t k = 0; k < n; ++k) {
        float a = patch0[k];
        float b = patch1[k];
        mean0 += a;
        mean1 += b;
        // ...
    }
    mean0 /= n;
    mean1 /= n;
    float csm = 0.0;
    float cm0 = 0.0;
    float cm1 = 0.0;
    for (size_t k = 0; k < n; ++k) {
        float ca = patch0[k] - mean0;
        float cb = patch1[k] - mean1;
        csm += ca * cb;
        cm0 += ca * ca;
        cm1 += cb * cb;
    }
    float zncc;
    if (cm0 == 0 || cm1 == 0)
        zncc = -1.0f;
    else
        zncc = csm / std::sqrt(cm0 * cm1);

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
    if (best_cost > 0.5f) return best_cost;

    float cost_sum = 0.0f;
    float cost_w = 0.0f;
    for (int i = 0; i < COSTS_BEST_K_LIMIT && i < costs.size(); ++i) {
        float cost = costs[i];
        if (cost > best_cost * COSTS_K_RATIO)
            break;
        float w = std::exp(-1.0 * i / COSTS_BEST_K_LIMIT);

        cost_sum += cost * w;
        cost_w += w;
    }
    // 110 реализуйте какое-то "усреднение cost-ов по всем соседям", с ограничением что участвуют только COSTS_BEST_K_LIMIT лучших
    // 111 добавьте к этому усреднению еще одно ограничение: если cost больше чем best_cost*COSTS_K_RATIO - то такой cost подозрительно плохой и мы его не хотим учитывать (вероятно occlusion)
    // 112 а что если в пикселе occlusion, но best_cost - большой и поэтому отсечение по best_cost*COSTS_K_RATIO не срабатывает? можно ли это отсечение как-то выправить для такого случая?
    // Мы знаем диапазон стоимостей [0, 1], поэтому можем понять, что cost большой и использовать другую формулу.
    // Например, с отрицательной корреляцией (cost > 0.5) как будто бы не стоит ничего усреднять 
    // TODO 207 а что если добавить какой-нибудь бонус в случае если больше чем Х камер засчиталось? улучшается/ухудшается ли от этого что-то на herzjezu25? а при большем числе фотографий

    float avg_cost = cost_sum / cost_w;
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

    verbose_cout << "hypothesis stats: ";
    for (auto hits : winner_hyp_hits) verbose_cout << 1.0 * hits / total_hyps << "\t";
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
