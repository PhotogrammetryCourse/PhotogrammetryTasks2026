#include "homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <set>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef ENABLE_ORSA_REFINE
#define ENABLE_ORSA_REFINE 1
#endif

namespace {
    constexpr double ORSA_precision_px_threshold = 5.;
    constexpr double ORSA_NFA_threshold = 1.;

    static_assert(ORSA_precision_px_threshold > 0.0, "ORSA_precision_px_threshold must be > 0");
    static_assert(ORSA_NFA_threshold > 0.0, "ORSA_NFA_threshold must be > 0");

    // источник: https://e-maxx.ru/algo/linear_systems_gauss
    // очень важно при выполнении метода гаусса использовать выбор опорного элемента: об этом можно почитать в источнике кода
    // или на вики: https://en.wikipedia.org/wiki/Pivot_element
    int gauss(std::vector<std::vector<double>> a, std::vector<double> &ans)
    {
        using namespace std;
        const double EPS = 1e-8;
        const int INF = std::numeric_limits<int>::max();

        int n = (int) a.size();
        int m = (int) a[0].size() - 1;

        vector<int> where (m, -1);
        for (int col=0, row=0; col<m && row<n; ++col) {
            int sel = row;
            for (int i=row; i<n; ++i)
                if (abs (a[i][col]) > abs (a[sel][col]))
                    sel = i;
            if (abs (a[sel][col]) < EPS)
                continue;
            for (int i=col; i<=m; ++i)
                swap (a[sel][i], a[row][i]);
            where[col] = row;

            for (int i=0; i<n; ++i)
                if (i != row) {
                    double c = a[i][col] / a[row][col];
                    for (int j=col; j<=m; ++j)
                        a[i][j] -= a[row][j] * c;
                }
            ++row;
        }

        ans.assign (m, 0);
        for (int i=0; i<m; ++i)
            if (where[i] != -1)
                ans[i] = a[where[i]][m] / a[where[i]][i];
        for (int i=0; i<n; ++i) {
            double sum = 0;
            for (int j=0; j<m; ++j)
                sum += ans[j] * a[i][j];
            if (abs (sum - a[i][m]) > EPS)
                return 0;
        }

        for (int i=0; i<m; ++i)
            if (where[i] == -1)
                return INF;
        return 1;
    }

    // см. Hartley, Zisserman: Multiple View Geometry in Computer Vision. Second Edition 4.1, 4.1.2
    cv::Mat estimateHomography4Points(const cv::Point2f &l0, const cv::Point2f &l1,
                                      const cv::Point2f &l2, const cv::Point2f &l3,
                                      const cv::Point2f &r0, const cv::Point2f &r1,
                                      const cv::Point2f &r2, const cv::Point2f &r3)
    {
        std::vector<std::vector<double>> A;
        std::vector<double> H;

        double xs0[4] = {l0.x, l1.x, l2.x, l3.x};
        double xs1[4] = {r0.x, r1.x, r2.x, r3.x};
        double ys0[4] = {l0.y, l1.y, l2.y, l3.y};
        double ys1[4] = {r0.y, r1.y, r2.y, r3.y};
        double ws0[4] = {1, 1, 1, 1};
        double ws1[4] = {1, 1, 1, 1};

        for (int i = 0; i < 4; ++i) {
            // fill 2 rows of matrix A

            double x0 = xs0[i];
            double y0 = ys0[i];
            double w0 = ws0[i];

            double x1 = xs1[i];
            double y1 = ys1[i];
            double w1 = ws1[i];

            // 8 elements of matrix + free term as needed by gauss routine
            A.push_back({x0, y0, w0, 0., 0., 0., -x1 * x0, -x1 * y0, x1 * w0});
            A.push_back({0., 0., 0., x0, y0, w0, -y1 * x0, -y1 * y0, y1 * w0});
        }

        int res = gauss(A, H);
        if (res == 0) {
            throw std::runtime_error("gauss: no solution found");
        }
        else
        if (res == 1) {
//            std::cout << "gauss: unique solution found" << std::endl;
        }
        else
        if (res == std::numeric_limits<int>::max()) {
            std::cerr << "gauss: infinitely many solutions found" << std::endl;
            std::cerr << "gauss: xs0: ";
            for (int i = 0; i < 4; ++i) {
                std::cerr << xs0[i] << ", ";
            }
            std::cerr << "\ngauss: ys0: ";
            for (int i = 0; i < 4; ++i) {
                std::cerr << ys0[i] << ", ";
            }
            std::cerr << std::endl;
        }
        else
        {
            throw std::runtime_error("gauss: unexpected return code");
        }

        // add fixed element H33 = 1
        H.push_back(1.0);

        cv::Mat H_mat(3, 3, CV_64FC1);
        std::copy(H.begin(), H.end(), H_mat.ptr<double>());
        return H_mat;
    }

    // pseudorandom number generator
    inline uint64_t xorshift64(uint64_t *state)
    {
        if (*state == 0) {
            *state = 1;
        }

        uint64_t x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return *state = x;
    }

    struct NormalizationData {
        double width = 1., height = 1.;
        cv::Matx33d T = cv::Matx33d::eye(), T_inv = cv::Matx33d::eye();
    };

    struct MatchSet {
        std::vector<cv::Point2d> lhs, rhs, lhs_norm, rhs_norm;
        NormalizationData lhs_normalization, rhs_normalization;
    };

    struct ModelEval {
        double log_nfa = std::numeric_limits<double>::infinity();
        int support = 0;
        std::vector<int> inliers;
    };

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    void removeDuplicateMatches(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs, std::vector<cv::Point2f> &unique_lhs, std::vector<cv::Point2f> &unique_rhs)
    {
        unique_lhs.clear();
        unique_rhs.clear();
        std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> was;
        for (int i = 0; i < points_lhs.size(); ++i) {
            const auto key = std::make_tuple(floatBits(points_lhs[i].x), floatBits(points_lhs[i].y), floatBits(points_rhs[i].x), floatBits(points_rhs[i].y));
            if (was.insert(key).second) {
                unique_lhs.push_back(points_lhs[i]);
                unique_rhs.push_back(points_rhs[i]);
            }
        }
    }

    void randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state)
    {
        dst.clear();

        const int max_attempts = 1000;

        for (int i = 0; i < sample_size; ++i) {
            for (int k = 0; k < max_attempts; ++k) {
                int v = xorshift64(state) % max_id;
                if (dst.empty() || std::find(dst.begin(), dst.end(), v) == dst.end()) {
                    dst.push_back(v);
                    break;
                }
            }
            if (dst.size() < i + 1) {
                throw std::runtime_error("Failed to sample ids");
            }
        }
    }

    void randomSampleFromPool(std::vector<int> &dst, const std::vector<int> &pool, int sample_size, uint64_t *state)
    {
        if (pool.size() < sample_size)
            throw std::runtime_error("Failed to sample ids");

        dst.clear();
        dst.reserve(sample_size);

        const int max_attempts = 1000;

        for (int i = 0; i < sample_size; ++i) {
            for (int k = 0; k < max_attempts; ++k) {
                int v = pool[xorshift64(state) % pool.size()];
                if (std::find(dst.begin(), dst.end(), v) == dst.end()) {
                    dst.push_back(v);
                    break;
                }
            }
            if (dst.size() < i + 1) {
                throw std::runtime_error("Failed to sample ids");
            }
        }
    }

    bool transformPoint(const cv::Point2d &pt, const cv::Matx33d &T, cv::Point2d &dst)
    {
        const double x = pt.x, y = pt.y;
        const double z = T(2, 0) * x + T(2, 1) * y + T(2, 2);
        if (std::abs(z) < 1e-8 || !std::isfinite(z))
            return false;

        dst.x = (T(0, 0) * x + T(0, 1) * y + T(0, 2)) / z;
        dst.y = (T(1, 0) * x + T(1, 1) * y + T(1, 2)) / z;
        return std::isfinite(dst.x) && std::isfinite(dst.y);
    }

    cv::Mat matFromMatx(const cv::Matx33d &T)
    {
        cv::Mat out(3, 3, CV_64F);
        std::copy(T.val, T.val + 9, out.ptr<double>());
        return out;
    }

    bool normalizeHomographyScale(cv::Matx33d &H)
    {
        double scale = H(2, 2);
        if (std::abs(scale) < 1e-8)
            scale = cv::norm(matFromMatx(H));

        if (std::abs(scale) < 1e-8 || !std::isfinite(scale))
            return false;

        H /= scale;
        for (const double& val : H.val) {
            if (!std::isfinite(val))
                return false;
        }
        return true;
    }

    NormalizationData makeNormalization(const std::vector<cv::Point2f> &points)
    {
        double max_x = 0., max_y = 0.;
        for (const cv::Point2f &pt : points) {
            max_x = std::max(max_x, std::ceil((double)pt.x));
            max_y = std::max(max_y, std::ceil((double)pt.y));
        }

        NormalizationData data;
        data.width = std::max(1., max_x + 1.);
        data.height = std::max(1., max_y + 1.);
        const double s = std::sqrt(data.width * data.height);
        data.T = cv::Matx33d(
            1./s, 0., -data.width/(2.*s),
            0., 1./s, -data.height/(2.*s),
            0., 0., 1.
        );
        data.T_inv = cv::Matx33d(
            s, 0., data.width/2.,
            0., s, data.height/2.,
            0., 0., 1.
        );
        return data;
    }

    bool satisfiesPositiveJacobian(const cv::Matx33d &H, const cv::Point2d &pt)
    {
        const double det_h = cv::determinant(matFromMatx(H));
        if (!std::isfinite(det_h) || std::abs(det_h) < 1e-8)
            return false;

        const double numerator = H(2, 0) * pt.x + H(2, 1) * pt.y + H(2, 2);
        return std::isfinite(numerator) && numerator / det_h > 0.;
    }

    bool estimateHomography(const MatchSet &matches, const std::vector<int> &indices, cv::Matx33d &H_norm, cv::Matx33d &H, const bool& demand_unique_solution_and_orientation_on_indices)
    {
        if (indices.size() < 4)
            return false;

        cv::Mat A = cv::Mat::zeros(2 * indices.size(), 9, CV_64F);
        for (int i = 0; i < indices.size(); ++i) {
            const int idx = indices[i];
            const cv::Point2d &lhs = matches.lhs_norm[idx];
            const cv::Point2d &rhs = matches.rhs_norm[idx];

            const int row0 = 2 * i;
            const int row1 = row0 + 1;

            A.at<double>(row0, 0) = lhs.x;
            A.at<double>(row0, 1) = lhs.y;
            A.at<double>(row0, 2) = 1.;
            A.at<double>(row0, 6) = -rhs.x * lhs.x;
            A.at<double>(row0, 7) = -rhs.x * lhs.y;
            A.at<double>(row0, 8) = -rhs.x;

            A.at<double>(row1, 3) = lhs.x;
            A.at<double>(row1, 4) = lhs.y;
            A.at<double>(row1, 5) = 1.;
            A.at<double>(row1, 6) = -rhs.y * lhs.x;
            A.at<double>(row1, 7) = -rhs.y * lhs.y;
            A.at<double>(row1, 8) = -rhs.y;
        }

        int svd_flags = cv::SVD::MODIFY_A;
        if (A.rows < A.cols)
            svd_flags |= cv::SVD::FULL_UV;

        cv::Mat singular_values, vt;
        cv::SVD::compute(A, singular_values, cv::noArray(), vt, svd_flags);

        if (demand_unique_solution_and_orientation_on_indices) {
            const int n = singular_values.total();
            const double sigma_max = singular_values.at<double>(0), sigma_prev = singular_values.at<double>(n - 2), sigma_last = singular_values.at<double>(n - 1);
            if ((sigma_last >= 0.01 * sigma_max) || (sigma_prev < 0.01 * sigma_max && sigma_last >= 0.1 * sigma_prev))
                return false;
        }

        cv::Mat h_vec = vt.row(vt.rows - 1).reshape(1, 3);
        H_norm = cv::Matx33d(
            h_vec.at<double>(0, 0), h_vec.at<double>(0, 1), h_vec.at<double>(0, 2),
            h_vec.at<double>(1, 0), h_vec.at<double>(1, 1), h_vec.at<double>(1, 2),
            h_vec.at<double>(2, 0), h_vec.at<double>(2, 1), h_vec.at<double>(2, 2)
        );

        H = matches.rhs_normalization.T_inv * H_norm * matches.lhs_normalization.T;
        if (!normalizeHomographyScale(H_norm) || !normalizeHomographyScale(H))
            return false;

        cv::Mat H_norm_singular_values;
        cv::SVD::compute(matFromMatx(H_norm), H_norm_singular_values, cv::SVD::NO_UV);
        const double sigma_max = H_norm_singular_values.at<double>(0, 0), sigma_min = H_norm_singular_values.at<double>(2, 0);
        if (sigma_min <= 1e-8 || (sigma_max / sigma_min) > 10.)
            return false;

        if (demand_unique_solution_and_orientation_on_indices) {
            for (const int& i : indices) {
                if (!satisfiesPositiveJacobian(H, matches.lhs[i]))
                    return false;
            }
        }

        return true;
    }

    bool evaluateHomographyACRansac(const MatchSet &matches, const cv::Matx33d &H_norm, const cv::Matx33d &H, const double& loge0, const std::vector<double> &logc_n, const std::vector<double> &logc_k, const double& max_error2, ModelEval &eval)
    {
        const int n_samples = 4;
        const int n_matches = matches.lhs.size();
        const double logalpha0 = std::log10(M_PI);

        std::vector<std::pair<double, int>> residuals;
        residuals.reserve(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            if (!satisfiesPositiveJacobian(H, matches.lhs[i])) {
                residuals.push_back({std::numeric_limits<double>::infinity(), i});
                continue;
            }

            cv::Point2d proj;
            if (!transformPoint(matches.lhs_norm[i], H_norm, proj)) {
                residuals.push_back({std::numeric_limits<double>::infinity(), i});
                continue;
            }

            const cv::Point2d diff = proj - matches.rhs_norm[i];
            residuals.push_back({diff.dot(diff), i});
        }
        std::sort(residuals.begin(), residuals.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first;
        });

        const int first_k = n_samples + 1;
        eval = {};
        for (int k = first_k; k <= n_matches; ++k) {
            const double error2 = residuals[k - 1].first;
            if (!std::isfinite(error2) || error2 > max_error2)
                break;

            const double logalpha = std::min(0., logalpha0 + std::log10(error2));
            const double log_nfa = loge0 + logc_n[k] + logc_k[k] + logalpha * (double)(k - n_samples);
            if (log_nfa < eval.log_nfa) {
                eval.log_nfa = log_nfa;
                eval.support = k;
            }
        }
        if (eval.support == 0)
            return false;

        eval.inliers.reserve(eval.support);
        for (int i = 0; i < eval.support; ++i)
            eval.inliers.push_back(residuals[i].second);

        return true;
    }

    bool refineHomographyFromInliers(const MatchSet &matches, const std::vector<int> &inliers, const double& loge0, const std::vector<double> &logc_n, const std::vector<double> &logc_k, const double& max_error2, cv::Matx33d &refined_H, ModelEval &eval)
    {
        cv::Matx33d refined_H_norm;
        if (!estimateHomography(matches, inliers, refined_H_norm, refined_H, false))
            return false;

        return evaluateHomographyACRansac(matches, refined_H_norm, refined_H, loge0, logc_n, logc_k, max_error2, eval);
    }

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
        }

        const int n_samples = 4;
        std::vector<cv::Point2f> unique_lhs, unique_rhs;
        removeDuplicateMatches(points_lhs, points_rhs, unique_lhs, unique_rhs);

        if (unique_lhs.size() < n_samples)
            throw std::runtime_error("findHomography : unique_lhs.size() < n_samples");

        MatchSet matches;
        matches.lhs_normalization = makeNormalization(unique_lhs);
        matches.rhs_normalization = makeNormalization(unique_rhs);

        matches.lhs.reserve(unique_lhs.size());
        matches.rhs.reserve(unique_rhs.size());
        matches.lhs_norm.reserve(unique_lhs.size());
        matches.rhs_norm.reserve(unique_rhs.size());
        for (int i = 0; i < unique_lhs.size(); ++i) {
            matches.lhs.emplace_back(unique_lhs[i]);
            matches.rhs.emplace_back(unique_rhs[i]);

            cv::Point2d lhs_norm, rhs_norm;
            transformPoint(matches.lhs.back(), matches.lhs_normalization.T, lhs_norm);
            transformPoint(matches.rhs.back(), matches.rhs_normalization.T, rhs_norm);
            matches.lhs_norm.push_back(lhs_norm);
            matches.rhs_norm.push_back(rhs_norm);
        }

        const int n_matches = matches.lhs.size();

        std::vector<int> all_indices(n_matches);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        if (n_matches == n_samples) {
            cv::Matx33d H, H_norm;
            if (!estimateHomography(matches, all_indices, H_norm, H, true))
                throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography from 4 points");

            return matFromMatx(H);
        }

        const int n_trials_total = 10000;
        int n_trials_reserve = n_trials_total / 10;
        int n_trials = n_trials_total - n_trials_reserve;
        const double loge0 = std::log10((double)std::max(1, n_matches - n_samples)), log_epsilon = std::log10(ORSA_NFA_threshold), max_precision_norm = ORSA_precision_px_threshold * matches.rhs_normalization.T(0, 0);
        const double max_error2 = max_precision_norm * max_precision_norm;

        std::vector<double> logc_n(n_matches + 1, 0.);
        std::vector<double> logc_k(n_matches + 1, 0.);
        for (int k = 0; k <= n_matches; ++k) {
            logc_n[k] = (std::lgamma(n_matches + 1.) - std::lgamma(k + 1.) - std::lgamma(n_matches - k + 1.)) / std::log(10.);
            if (k >= n_samples)
                logc_k[k] = (std::lgamma(k + 1.) - std::lgamma(n_samples + 1.) - std::lgamma(k - n_samples + 1.)) / std::log(10.);
        }

        uint64_t seed = 1;
        std::vector<int> sampling_pool = all_indices, sample, best_inliers;
        int best_support = 0;
        double best_log_nfa = std::numeric_limits<double>::infinity();
        cv::Matx33d best_H = cv::Matx33d::eye();
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            randomSampleFromPool(sample, sampling_pool, n_samples, &seed);

            cv::Matx33d H_norm;
            cv::Matx33d H;
            if (!estimateHomography(matches, sample, H_norm, H, true))
                continue;

            ModelEval cand;
            if (!evaluateHomographyACRansac(matches, H_norm, H, loge0, logc_n, logc_k, max_error2, cand))
                continue;

            const bool cand_better = (cand.log_nfa < best_log_nfa || (std::abs(cand.log_nfa - best_log_nfa) < 1e-8 && cand.support > best_support));
            if (cand_better) {
                best_log_nfa = cand.log_nfa;
                best_support = cand.support;
                best_H = H;
                best_inliers = cand.inliers;
            }

            if (cand_better && cand.log_nfa < log_epsilon) {
                sampling_pool = best_inliers;
                if (n_trials_reserve > 0) {
                    n_trials = (i_trial + 1) + n_trials_reserve;
                    n_trials_reserve = 0;
                }
            } else if (i_trial + 1 == n_trials && n_trials_reserve > 0 && !best_inliers.empty()) {
                sampling_pool = best_inliers;
                n_trials = (i_trial + 1) + n_trials_reserve;
                n_trials_reserve = 0;
            }
        }
        if (best_log_nfa >= log_epsilon || best_inliers.empty()) {
            throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography");
        }

#if ENABLE_ORSA_REFINE
        std::vector<int> current_inliers = best_inliers;
        cv::Matx33d current_H = best_H;
        bool refined_successfully = false;
        for (int i = 0; i < 10; ++i) {  // it just safer than while (true)
            cv::Matx33d refined_H;
            ModelEval refined;
            if (!refineHomographyFromInliers(matches, current_inliers, loge0, logc_n, logc_k, max_error2, refined_H, refined))
                break;

            current_H = refined_H;
            refined_successfully = true;
            if (refined.inliers == current_inliers)
                break;

            current_inliers.swap(refined.inliers);
        }
        if (refined_successfully)
            best_H = current_H;
#else
        cv::Matx33d refined_H;
        ModelEval refined;
        if (refineHomographyFromInliers(matches, best_inliers, loge0, logc_n, logc_k, max_error2, refined_H, refined))
            best_H = refined_H;
#endif

        return matFromMatx(best_H);
    }

}

cv::Mat phg::findHomography(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return estimateHomographyRANSAC(points_lhs, points_rhs);
}

// чтобы заработало, нужно пересобрать библиотеку с дополнительным модулем calib3d (см. инструкцию в корневом CMakeLists.txt)
cv::Mat phg::findHomographyCV(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return cv::findHomography(points_lhs, points_rhs, cv::RANSAC);
}

// T - 3x3 однородная матрица, например, гомография
// таким преобразованием внутри занимается функции cv::perspectiveTransform и cv::warpPerspective
cv::Point2d phg::transformPoint(const cv::Point2d &pt, const cv::Mat &T)
{
    cv::Mat Td;
    T.convertTo(Td, CV_64F);

    const double x = pt.x, y = pt.y;
    const double z = Td.at<double>(2, 0) * x + Td.at<double>(2, 1) * y + Td.at<double>(2, 2);
    if (std::abs(z) < 1e-8)
        throw std::runtime_error("transformPoint: unacceptably small scale");

    return {(Td.at<double>(0, 0) * x + Td.at<double>(0, 1) * y + Td.at<double>(0, 2)) / z, (Td.at<double>(1, 0) * x + Td.at<double>(1, 1) * y + Td.at<double>(1, 2)) / z};
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
