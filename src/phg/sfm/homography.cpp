#include "homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

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
    cv::Mat estimateHomographyLeastSquares(const std::vector<cv::Point2f> &points_lhs,
                                           const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("estimateHomographyLeastSquares: points_lhs.size() != points_rhs.size()");
        }
        if (points_lhs.size() < 4) {
            throw std::runtime_error("estimateHomographyLeastSquares: too few points");
        }

        const int n_points = static_cast<int>(points_lhs.size());
        cv::Mat A(2 * n_points, 8, CV_64FC1);
        cv::Mat b(2 * n_points, 1, CV_64FC1);

        for (int i = 0; i < n_points; ++i) {
            const double x0 = points_lhs[i].x;
            const double y0 = points_lhs[i].y;
            const double x1 = points_rhs[i].x;
            const double y1 = points_rhs[i].y;

            A.at<double>(2 * i + 0, 0) = x0;
            A.at<double>(2 * i + 0, 1) = y0;
            A.at<double>(2 * i + 0, 2) = 1.0;
            A.at<double>(2 * i + 0, 3) = 0.0;
            A.at<double>(2 * i + 0, 4) = 0.0;
            A.at<double>(2 * i + 0, 5) = 0.0;
            A.at<double>(2 * i + 0, 6) = -x0 * x1;
            A.at<double>(2 * i + 0, 7) = -y0 * x1;
            b.at<double>(2 * i + 0, 0) = x1;

            A.at<double>(2 * i + 1, 0) = 0.0;
            A.at<double>(2 * i + 1, 1) = 0.0;
            A.at<double>(2 * i + 1, 2) = 0.0;
            A.at<double>(2 * i + 1, 3) = x0;
            A.at<double>(2 * i + 1, 4) = y0;
            A.at<double>(2 * i + 1, 5) = 1.0;
            A.at<double>(2 * i + 1, 6) = -x0 * y1;
            A.at<double>(2 * i + 1, 7) = -y0 * y1;
            b.at<double>(2 * i + 1, 0) = y1;
        }

        cv::Mat h;
        if (!cv::solve(A, b, h, cv::DECOMP_SVD)) {
            throw std::runtime_error("estimateHomographyLeastSquares: cv::solve failed");
        }

        cv::Mat H(3, 3, CV_64FC1);
        H.at<double>(0, 0) = h.at<double>(0, 0);
        H.at<double>(0, 1) = h.at<double>(1, 0);
        H.at<double>(0, 2) = h.at<double>(2, 0);
        H.at<double>(1, 0) = h.at<double>(3, 0);
        H.at<double>(1, 1) = h.at<double>(4, 0);
        H.at<double>(1, 2) = h.at<double>(5, 0);
        H.at<double>(2, 0) = h.at<double>(6, 0);
        H.at<double>(2, 1) = h.at<double>(7, 0);
        H.at<double>(2, 2) = 1.0;
        return H;
    }

    double logBinomialCoeff(int n, int k)
    {
        if (k < 0 || k > n) {
            return -std::numeric_limits<double>::infinity();
        }
        return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
    }

    double estimateBackgroundArea(const std::vector<cv::Point2f> &points)
    {
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        for (const cv::Point2f &pt : points) {
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
        }

        const double width = std::max(1.0, double(max_x - min_x + 1.0f));
        const double height = std::max(1.0, double(max_y - min_y + 1.0f));
        return width * height;
    }

    double computeLogNFA(const std::vector<double> &sorted_errors,
                         int k,
                         int sample_size,
                         double alpha0)
    {
        const double eps = std::max(1e-12, sorted_errors[k - 1]);
        const double log_p = std::min(0.0, 2.0 * std::log(eps) + std::log(alpha0));
        return std::log(double(sorted_errors.size() - sample_size)) +
               logBinomialCoeff(static_cast<int>(sorted_errors.size()), k) +
               logBinomialCoeff(k, sample_size) +
               (k - sample_size) * log_p;
    }

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
            A.push_back({x0, y0, w0, 0.0, 0.0, 0.0, -x0 * x1 / w1, -y0 * x1 / w1, x1 / w1});
            A.push_back({0.0, 0.0, 0.0, x0, y0, w0, -x0 * y1 / w1, -y0 * y1 / w1, y1 / w1});
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
            throw std::runtime_error("gauss: degenerate 4-point sample");
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

    int updateRequiredTrials(int current_limit, int n_matches, int n_inliers, int sample_size)
    {
        const int min_trials = 128;

        if (n_inliers < sample_size) {
            return current_limit;
        }

        const double confidence = 0.999;
        const double inlier_ratio = std::clamp(double(n_inliers) / double(n_matches), 1e-12, 1.0 - 1e-12);
        const double all_inliers_prob = std::clamp(std::pow(inlier_ratio, sample_size), 1e-12, 1.0 - 1e-12);
        const int required = int(std::ceil(std::log(1.0 - confidence) / std::log(1.0 - all_inliers_prob)));
        return std::max(min_trials, std::min(current_limit, required));
    }

    bool evaluateHomographyAContrario(const cv::Mat &H,
                                      const std::vector<cv::Point2f> &points_lhs,
                                      const std::vector<cv::Point2f> &points_rhs,
                                      int sample_size,
                                      double alpha0,
                                      std::vector<std::pair<double, int>> &residuals_with_idx,
                                      std::vector<double> &sorted_errors,
                                      double &best_log_nfa,
                                      int &best_k)
    {
        const int n_matches = static_cast<int>(points_lhs.size());
        for (int i = 0; i < n_matches; ++i) {
            const cv::Point2d proj = phg::transformPoint(points_lhs[i], H);
            const double err = cv::norm(proj - cv::Point2d(points_rhs[i]));
            residuals_with_idx[i] = {err, i};
        }

        std::sort(residuals_with_idx.begin(), residuals_with_idx.end(),
                  [](const std::pair<double, int> &lhs, const std::pair<double, int> &rhs) {
                      return lhs.first < rhs.first;
                  });

        for (int i = 0; i < n_matches; ++i) {
            sorted_errors[i] = residuals_with_idx[i].first;
        }

        best_log_nfa = std::numeric_limits<double>::infinity();
        best_k = -1;
        for (int k = sample_size + 1; k <= n_matches; ++k) {
            const double log_nfa = computeLogNFA(sorted_errors, k, sample_size, alpha0);
            if (log_nfa < best_log_nfa) {
                best_log_nfa = log_nfa;
                best_k = k;
            }
        }

        return best_k >= sample_size + 1 && std::isfinite(best_log_nfa);
    }

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
        }
        if (points_lhs.size() < 4) {
            throw std::runtime_error("findHomography: too few correspondences");
        }

        // TODO Дополнительный балл, если вместо обычной версии будет использована модификация a-contrario RANSAC
        // * [1] Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers. (Lionel Moisan, Pierre Moulon, Pascal Monasse)
        // * [2] Adaptive Structure from Motion with a contrario model estimation. (Pierre Moulon, Pascal Monasse, Renaud Marlet)
        // * (простое описание для понимания)
        // * [3] http://ikrisoft.blogspot.com/2015/01/ransac-with-contrario-approach.html

        const int n_matches = static_cast<int>(points_lhs.size());
        const int max_trials = 5000;
        const int n_samples = 4;
        const double alpha0 = M_PI / std::max(4.0 * M_PI, estimateBackgroundArea(points_rhs));

        double best_log_nfa = std::numeric_limits<double>::infinity();
        int best_k = 0;
        cv::Mat best_H;
        std::vector<int> best_inliers;

        uint64_t seed = 1;
        std::vector<int> sample;
        std::vector<std::pair<double, int>> residuals_with_idx(n_matches);
        std::vector<double> sorted_errors(n_matches);
        int required_trials = max_trials;

        for (int i_trial = 0; i_trial < required_trials; ++i_trial) {
            randomSample(sample, n_matches, n_samples, &seed);

            cv::Mat H;
            try {
                H = estimateHomography4Points(points_lhs[sample[0]], points_lhs[sample[1]], points_lhs[sample[2]], points_lhs[sample[3]],
                                              points_rhs[sample[0]], points_rhs[sample[1]], points_rhs[sample[2]], points_rhs[sample[3]]);
            } catch (const std::exception &) {
                continue;
            }

            double model_log_nfa;
            int model_k;
            try {
                if (!evaluateHomographyAContrario(H, points_lhs, points_rhs, n_samples, alpha0,
                                                  residuals_with_idx, sorted_errors, model_log_nfa, model_k)) {
                    continue;
                }
            } catch (const std::exception &) {
                continue;
            }

            if (model_log_nfa < best_log_nfa ||
                (model_log_nfa == best_log_nfa && model_k > best_k)) {
                best_log_nfa = model_log_nfa;
                best_k = model_k;
                best_H = H;
                best_inliers.clear();
                best_inliers.reserve(model_k);
                for (int i = 0; i < model_k; ++i) {
                    best_inliers.push_back(residuals_with_idx[i].second);
                }
                required_trials = updateRequiredTrials(required_trials, n_matches, model_k, n_samples);
            }
            if (best_k == n_matches) {
                break;
            }
        }

        if (best_H.empty() || best_k < n_samples + 1) {
            throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography");
        }

        cv::Mat refined_H = best_H;
        std::vector<int> refined_inliers = best_inliers;
        double refined_log_nfa = best_log_nfa;

        for (int i_refine = 0; i_refine < 2; ++i_refine) {
            if (refined_inliers.size() < n_samples) {
                break;
            }

            std::vector<cv::Point2f> inliers_lhs;
            std::vector<cv::Point2f> inliers_rhs;
            inliers_lhs.reserve(refined_inliers.size());
            inliers_rhs.reserve(refined_inliers.size());
            for (int idx : refined_inliers) {
                inliers_lhs.push_back(points_lhs[idx]);
                inliers_rhs.push_back(points_rhs[idx]);
            }

            cv::Mat candidate_H;
            try {
                candidate_H = estimateHomographyLeastSquares(inliers_lhs, inliers_rhs);
            } catch (const std::exception &) {
                break;
            }

            double candidate_log_nfa;
            int candidate_k;
            try {
                if (!evaluateHomographyAContrario(candidate_H, points_lhs, points_rhs, n_samples, alpha0,
                                                  residuals_with_idx, sorted_errors, candidate_log_nfa, candidate_k)) {
                    break;
                }
            } catch (const std::exception &) {
                break;
            }

            if (candidate_log_nfa >= refined_log_nfa &&
                candidate_k <= static_cast<int>(refined_inliers.size())) {
                break;
            }

            refined_H = candidate_H;
            refined_log_nfa = candidate_log_nfa;
            refined_inliers.clear();
            refined_inliers.reserve(candidate_k);
            for (int i = 0; i < candidate_k; ++i) {
                refined_inliers.push_back(residuals_with_idx[i].second);
            }
        }

        return refined_H;
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
    if (T.rows != 3 || T.cols != 3) {
        throw std::runtime_error("transformPoint: expected 3x3 matrix");
    }

    const double x = pt.x;
    const double y = pt.y;

    const double tx = T.at<double>(0, 0) * x + T.at<double>(0, 1) * y + T.at<double>(0, 2);
    const double ty = T.at<double>(1, 0) * x + T.at<double>(1, 1) * y + T.at<double>(1, 2);
    const double tw = T.at<double>(2, 0) * x + T.at<double>(2, 1) * y + T.at<double>(2, 2);

    if (std::abs(tw) < 1e-12) {
        throw std::runtime_error("transformPoint: homogeneous coordinate is too small");
    }

    return cv::Point2d(tx / tw, ty / tw);
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
