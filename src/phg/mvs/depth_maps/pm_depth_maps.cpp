#include "pm_depth_maps.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <libutils/fast_random.h>
#include <libutils/timer.h>
#include <phg/utils/point_cloud_export.h>

#include "pm_depth_maps_defines.h"
#include "pm_fast_random.h"
#include "pm_geometry.h"

namespace phg {
namespace {

vector3d cameraCenter(const matrix34d& PtoWorld)
{
    return vector3d(PtoWorld(0, 3), PtoWorld(1, 3), PtoWorld(2, 3));
}

vector3f normalizeSafe(const vector3f& v, const vector3f& fallback)
{
    float len2 = norm2(v);
    if (len2 <= 1e-12f) {
        float fallback_len2 = norm2(fallback);
        if (fallback_len2 <= 1e-12f) {
            return vector3f(0.0f, 0.0f, -1.0f);
        }
        return fallback * (1.0f / std::sqrt(fallback_len2));
    }
    return v * (1.0f / std::sqrt(len2));
}

int makeOddAtLeast(int value, int min_value)
{
    value = std::max(value, min_value);
    if (value % 2 == 0) {
        ++value;
    }
    return value;
}

bool sampleBilinearGrey(const cv::Mat& img, double x, double y, float& intensity)
{
    if (!(x >= 0.5 && x <= img.cols - 0.5 && y >= 0.5 && y <= img.rows - 0.5)) {
        return false;
    }

    double fx = x - 0.5;
    double fy = y - 0.5;

    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));

    x0 = std::max(0, std::min(x0, img.cols - 1));
    y0 = std::max(0, std::min(y0, img.rows - 1));

    int x1 = std::min(x0 + 1, img.cols - 1);
    int y1 = std::min(y0 + 1, img.rows - 1);

    double ax = fx - x0;
    double ay = fy - y0;

    float i00 = img.at<unsigned char>(y0, x0) / 255.0f;
    float i10 = img.at<unsigned char>(y0, x1) / 255.0f;
    float i01 = img.at<unsigned char>(y1, x0) / 255.0f;
    float i11 = img.at<unsigned char>(y1, x1) / 255.0f;

    intensity = static_cast<float>((1.0 - ax) * (1.0 - ay) * i00
        + ax * (1.0 - ay) * i10
        + (1.0 - ax) * ay * i01
        + ax * ay * i11);
    return true;
}

} // namespace

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
    pixel_with_depth[2] = depth;

    return pixel_with_depth;
}

vector3d unproject(const vector3d& pixel, const phg::Calibration& calibration, const matrix34d& PtoWorld)
{
    double depth = pixel[2];

    vector3d ray_local = calibration.unproject(vector2d(pixel[0], pixel[1]));
    vector3d local_point = ray_local * depth;

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

    depth_map = cv::Mat::zeros(height, width, CV_32FC1);
    normal_map = cv::Mat::zeros(height, width, CV_32FC3);
    cost_map = cv::Mat::zeros(height, width, CV_32FC1);

    iter = 0;

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
            FastRandom r(iter, j * width + i);

            float d0, dp, dr;
            vector3f n0, np, nr;

            {
                float progress = (NITERATIONS > 0) ? static_cast<float>(iter) / static_cast<float>(NITERATIONS) : 1.0f;
                float depth_span = ref_depth_max - ref_depth_min;
                float depth_rel_radius = std::max(0.03f, 0.5f * (1.0f - progress) + 0.05f);
                float depth_abs_radius = depth_span * 0.02f;
                float normal_radius = std::max(0.05f, 0.5f * (1.0f - progress) + 0.05f);

                d0 = depth_map.at<float>(j, i);
                n0 = normal_map.at<vector3f>(j, i);

                if (d0 == NO_DEPTH || !(d0 >= ref_depth_min && d0 <= ref_depth_max) || norm2(n0) < 0.5f) {
                    d0 = r.nextf(ref_depth_min, ref_depth_max);
                    n0 = randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r);
                }
                n0 = normalizeSafe(n0, randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r));

                depth_abs_radius = std::max(depth_span * 0.02f, d0 * depth_rel_radius);
                float dp_min = std::max(ref_depth_min, d0 - depth_abs_radius);
                float dp_max = std::min(ref_depth_max, d0 + depth_abs_radius);
                dp = (dp_min < dp_max) ? r.nextf(dp_min, dp_max) : d0;
                np = normalizeSafe(n0 + randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r) * normal_radius, n0);

                dr = r.nextf(ref_depth_min, ref_depth_max);
                nr = randomNormalObservedFromCamera(cameras_RtoWorld[ref_cam], r);
            }

            float best_depth = depth_map.at<float>(j, i);
            vector3f best_normal = normal_map.at<vector3f>(j, i);
            float best_cost = cost_map.at<float>(j, i);
            if (best_depth == NO_DEPTH) {
                best_cost = NO_COST;
            }

            float depths[3] = { d0, dr, dp };
            vector3f normals[3] = { n0, nr, np };

            for (size_t hi = 0; hi < 3 * 3; ++hi) {
                float d = depths[hi / 3];
                vector3f n = normals[hi % 3];

                std::vector<float> costs;
                costs.reserve(ncameras > 0 ? ncameras - 1 : 0);
                for (size_t ni = 0; ni < ncameras; ++ni) {
                    if (ni == ref_cam)
                        continue;

                    float costi = estimateCost(i, j, d, n, ni);
                    costs.push_back(costi);
                }

                float total_cost = avgCost(costs);

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

    verbose_cout << "refinement done in " << t.elapsed() << " s: ";
#ifdef VERBOSE_LOGGING
    printCurrentStats();
#endif
#ifdef DEBUG_DIR
    debugCurrentPoints(std::to_string(ref_cam) + "_" + std::to_string(iter) + "_refinement");
#endif
}

void PMDepthMapsBuilder::tryToPropagateDonor(ptrdiff_t ni, ptrdiff_t nj, int chessboard_pattern_step, std::vector<float>& hypos_depth, std::vector<vector3f>& hypos_normal, std::vector<float>& hypos_cost)
{
    rassert((ni + nj) % 2 != chessboard_pattern_step, 2391249129510120);

    if (ni < 0 || ni >= width || nj < 0 || nj >= height)
        return;

    float d = depth_map.at<float>(nj, ni);
    if (d == NO_DEPTH)
        return;

    vector3f n = normal_map.at<vector3f>(nj, ni);
    float cost = cost_map.at<float>(nj, ni);

    hypos_depth.push_back(d);
    hypos_normal.push_back(n);
    hypos_cost.push_back(cost);
}

void PMDepthMapsBuilder::propagation()
{
    timer t;
    verbose_cout << "Iteration #" << iter << "/" << NITERATIONS << ": propagation..." << std::endl;

    int adaptive_far_step = PROPAGATION_STEP;
    if (iter > 0 && NITERATIONS > 0) {
        float progress = static_cast<float>(iter) / static_cast<float>(NITERATIONS);
        adaptive_far_step = static_cast<int>(std::round(PROPAGATION_STEP * (1.0f - 0.6f * progress)));
    }
    adaptive_far_step = makeOddAtLeast(adaptive_far_step, 3);

    int adaptive_mid_step = makeOddAtLeast(std::max(3, adaptive_far_step / 2), 3);
    if (adaptive_mid_step >= adaptive_far_step) {
        adaptive_mid_step = makeOddAtLeast(std::max(3, adaptive_far_step - 2), 3);
    }

    for (int chessboard_pattern_step = 0; chessboard_pattern_step < 2; ++chessboard_pattern_step) {
#pragma omp parallel for schedule(dynamic, 1)
        for (ptrdiff_t j = 0; j < height; ++j) {
            for (ptrdiff_t i = (j + chessboard_pattern_step) % 2; i < width; i += 2) {
                std::vector<float> hypos_depth;
                std::vector<vector3f> hypos_normal;
                std::vector<float> hypos_cost;
                hypos_depth.reserve(24);
                hypos_normal.reserve(24);
                hypos_cost.reserve(24);

                tryToPropagateDonor(i - 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 1, j + 0, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                tryToPropagateDonor(i - 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i - 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 1, j - 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 2, j - 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i - 1, j + 2, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i - 2, j + 1, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                tryToPropagateDonor(i - 1 * adaptive_mid_step, j + 0 * adaptive_mid_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0 * adaptive_mid_step, j - 1 * adaptive_mid_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 1 * adaptive_mid_step, j + 0 * adaptive_mid_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0 * adaptive_mid_step, j + 1 * adaptive_mid_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                tryToPropagateDonor(i - 1 * adaptive_far_step, j + 0 * adaptive_far_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0 * adaptive_far_step, j - 1 * adaptive_far_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 1 * adaptive_far_step, j + 0 * adaptive_far_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);
                tryToPropagateDonor(i + 0 * adaptive_far_step, j + 1 * adaptive_far_step, chessboard_pattern_step, hypos_depth, hypos_normal, hypos_cost);

                float best_depth = depth_map.at<float>(j, i);
                vector3f best_normal = normal_map.at<vector3f>(j, i);
                float best_cost = cost_map.at<float>(j, i);
                if (best_depth == NO_DEPTH) {
                    best_cost = NO_COST;
                }

                std::vector<size_t> order(hypos_depth.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
                    return hypos_cost[lhs] < hypos_cost[rhs];
                });

                size_t donors_limit = std::min<size_t>(8, order.size());
                for (size_t oi = 0; oi < donors_limit; ++oi) {
                    size_t hi = order[oi];

                    float d = hypos_depth[hi];
                    vector3f n = hypos_normal[hi];

                    std::vector<float> costs;
                    costs.reserve(ncameras > 0 ? ncameras - 1 : 0);
                    for (size_t ni = 0; ni < ncameras; ++ni) {
                        if (ni == ref_cam)
                            continue;

                        float costi = estimateCost(i, j, d, n, ni);
                        costs.push_back(costi);
                    }

                    float total_cost = avgCost(costs);

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
    debugCurrentPoints(std::to_string(ref_cam) + "_" + std::to_string(iter) + "_propagation");
#endif
}

float PMDepthMapsBuilder::estimateCost(ptrdiff_t i, ptrdiff_t j, double d, const vector3d& global_normal, size_t neighb_cam)
{
    if (!(d > 0.0) || d < ref_depth_min || d > ref_depth_max)
        return NO_COST;
    if (!(i - COST_PATCH_RADIUS >= 0 && i + COST_PATCH_RADIUS < width))
        return NO_COST;
    if (!(j - COST_PATCH_RADIUS >= 0 && j + COST_PATCH_RADIUS < height))
        return NO_COST;

    vector3d pixel(i + 0.5, j + 0.5, d);
    vector3d global_point = unproject(pixel, calibration, cameras_PtoWorld[ref_cam]);

    vector3d ref_camera_center = cameraCenter(cameras_PtoWorld[ref_cam]);
    if (dot(global_normal, ref_camera_center - global_point) <= 1e-9)
        return NO_COST;

    std::vector<float> patch0;
    std::vector<float> patch1;
    patch0.reserve((2 * COST_PATCH_RADIUS + 1) * (2 * COST_PATCH_RADIUS + 1));
    patch1.reserve((2 * COST_PATCH_RADIUS + 1) * (2 * COST_PATCH_RADIUS + 1));

    for (ptrdiff_t dj = -COST_PATCH_RADIUS; dj <= COST_PATCH_RADIUS; ++dj) {
        for (ptrdiff_t di = -COST_PATCH_RADIUS; di <= COST_PATCH_RADIUS; ++di) {
            ptrdiff_t ni = i + di;
            ptrdiff_t nj = j + dj;

            patch0.push_back(cameras_imgs_grey[ref_cam].at<unsigned char>(nj, ni) / 255.0f);

            vector3d local_ray = calibration.unproject(vector2d(ni + 0.5, nj + 0.5));
            vector3d ray_dir = cameras_RtoWorld[ref_cam] * local_ray;
            double ray_len = std::sqrt(dot(ray_dir, ray_dir));
            if (ray_len <= 1e-12)
                return NO_COST;
            ray_dir *= 1.0 / ray_len;

            vector3d global_intersection;
            if (!intersectWithPlane(global_point, global_normal, ref_camera_center, ray_dir, global_intersection))
                return NO_COST;

            rassert(neighb_cam != ref_cam, 2334195412410286);
            vector3d neighb_proj = project(global_intersection, calibration, cameras_PtoLocal[neighb_cam]);
            if (neighb_proj[2] <= 0.0)
                return NO_COST;

            double x = neighb_proj[0];
            double y = neighb_proj[1];

            float intensity = 0.0f;
            if (!sampleBilinearGrey(cameras_imgs_grey[neighb_cam], x, y, intensity))
                return NO_COST;

            patch1.push_back(intensity);
        }
    }

    rassert(patch0.size() == patch1.size(), 12489185129326);
    size_t n = patch0.size();
    if (n == 0)
        return NO_COST;

    float mean0 = 0.0f;
    float mean1 = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        mean0 += patch0[k];
        mean1 += patch1[k];
    }
    mean0 /= static_cast<float>(n);
    mean1 /= static_cast<float>(n);

    float num = 0.0f;
    float den0 = 0.0f;
    float den1 = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        float a = patch0[k] - mean0;
        float b = patch1[k] - mean1;
        num += a * b;
        den0 += a * a;
        den1 += b * b;
    }

    float zncc = 0.0f;
    if (den0 > 1e-12f && den1 > 1e-12f) {
        zncc = num / std::sqrt(den0 * den1);
    }

    rassert(zncc == zncc, 23141241210380);
    zncc = std::max(-1.0f, std::min(1.0f, zncc));
    rassert(zncc >= -1.0 && zncc <= 1.0, 141251251541357);

    float cost = (1.0f - zncc) / 2.0f;
    rassert(cost >= 0.0f && cost <= NO_COST, 23123912049102361);

    return cost;
}

float PMDepthMapsBuilder::avgCost(std::vector<float>& costs)
{
    std::vector<float> valid_costs;
    valid_costs.reserve(costs.size());
    for (float c : costs) {
        if (c >= 0.0f && c < NO_COST) {
            valid_costs.push_back(c);
        }
    }

    if (valid_costs.empty())
        return NO_COST;

    std::sort(valid_costs.begin(), valid_costs.end());

    float best_cost = valid_costs[0];
    float max_allowed_cost = std::max(best_cost, std::max(best_cost * COSTS_K_RATIO, best_cost + 0.05f));
    max_allowed_cost = std::min(0.95f, max_allowed_cost);

    float cost_sum = 0.0f;
    float cost_w = 0.0f;
    size_t used = 0;
    for (float c : valid_costs) {
        if (used >= COSTS_BEST_K_LIMIT)
            break;
        if (used > 0 && c > max_allowed_cost)
            break;

        cost_sum += c;
        cost_w += 1.0f;
        ++used;
    }

    if (cost_w <= 0.0f)
        return NO_COST;

    float avg_cost = cost_sum / cost_w;
    avg_cost = std::max(0.0f, std::min(NO_COST, avg_cost));
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
    double avg_cost = (costs_n > 0.0) ? (costs_sum / costs_n) : NO_COST;
    double avg_good_cost = (good_costs_n > 0.0) ? (good_costs_sum / good_costs_n) : NO_COST;
    verbose_cout << to_percent(costs_n, ntotal) << "% pixels with " << avg_cost << " avg cost, ";
    verbose_cout << to_percent(good_costs_n, ntotal) << "% pixels with good " << avg_good_cost << " avg cost";
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