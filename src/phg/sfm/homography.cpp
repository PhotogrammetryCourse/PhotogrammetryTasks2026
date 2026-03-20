#include "homography.h"
#include <cmath>
#include <opencv2/calib3d/calib3d.hpp>
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
            A.push_back({-x0, -y0, -1, 0, 0, 0, x0*x1, y0*x1, -x1});
            A.push_back({0, 0, 0, -x0, -y0, -1, x0*y1, y0*y1, -y1});
            
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

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    if (points_lhs.size() != points_rhs.size()) {
        throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
    }

    const int n_matches = points_lhs.size();
    if (n_matches < 4) {
        throw std::runtime_error("estimateHomographyRANSAC: not enough points");
    }

    if (n_matches == 4) {
        return estimateHomography4Points(points_lhs[0], points_lhs[1], points_lhs[2], points_lhs[3],
                                         points_rhs[0], points_rhs[1], points_rhs[2], points_rhs[3]);
    }

    double min_x = points_rhs[0].x, max_x = points_rhs[0].x;
    double min_y = points_rhs[0].y, max_y = points_rhs[0].y;
    for (const auto& p : points_rhs) {
        min_x = std::min(min_x, (double)p.x); max_x = std::max(max_x, (double)p.x);
        min_y = std::min(min_y, (double)p.y); max_y = std::max(max_y, (double)p.y);
    }
    double target_area = (max_x - min_x) * (max_y - min_y);
    if (target_area < 1.0) target_area = 1e6;

    const int n_trials = 5000;
    const int n_samples = 4;
    uint64_t seed = 1;

    double best_log_nfa = std::numeric_limits<double>::max();
    cv::Mat best_H;
    int best_support = 0;

    double log_N_minus_4 = std::log(std::max(1, n_matches - 4));
    double lgamma_n_plus_1 = std::lgamma(n_matches + 1.0);
    double lgamma_4_plus_1 = std::lgamma(n_samples + 1.0);

    std::vector<int> sample;
    std::vector<double> errors(n_matches);

    for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
        randomSample(sample, n_matches, n_samples, &seed);

        cv::Mat H;
        try {
            H = estimateHomography4Points(points_lhs[sample[0]], points_lhs[sample[1]], points_lhs[sample[2]], points_lhs[sample[3]],
                                          points_rhs[sample[0]], points_rhs[sample[1]], points_rhs[sample[2]], points_rhs[sample[3]]);
        } catch (...) {
            continue;
        }

        for (int i = 0; i < n_matches; ++i) {
            try {
                cv::Point2d proj = phg::transformPoint(points_lhs[i], H);
                cv::Point2d diff = proj - cv::Point2d(points_rhs[i]);
                errors[i] = diff.x * diff.x + diff.y * diff.y;
            } catch (...) {
                errors[i] = std::numeric_limits<double>::max();
            }
        }

        std::vector<double> sorted_errors = errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());

        for (int k = 5; k <= n_matches; ++k) {
            double e2 = sorted_errors[k - 1];
            
            if (e2 >= std::numeric_limits<double>::max() / 2.0) break;
            
            double alpha = (CV_PI * e2) / target_area;
            if (alpha >= 1.0) continue;
            alpha = std::max(alpha, 1e-12);

            double lgamma_k_plus_1 = std::lgamma(k + 1.0);
            double log_binom_n_k = lgamma_n_plus_1 - lgamma_k_plus_1 - std::lgamma(n_matches - k + 1.0);
            double log_binom_k_4 = lgamma_k_plus_1 - lgamma_4_plus_1 - std::lgamma(k - n_samples + 1.0);

            double log_nfa = log_N_minus_4 + log_binom_n_k + log_binom_k_4 + (k - n_samples) * std::log(alpha);

            if (log_nfa < best_log_nfa) {
                best_log_nfa = log_nfa;
                best_H = H;
                best_support = k;
            }
        }
    }

    if (best_log_nfa > 0) {
        throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography");
    }

    std::cout << "estimateHomographyRANSAC : support: " << best_support << "/" << n_matches << std::endl;

    return best_H;
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
    double x = pt.x, y = pt.y;
    double w = T.at<double>(2,0) * x + T.at<double>(2,1) * y + T.at<double>(2,2);
    
    if (std::abs(w) < 1e-8) {
        throw std::runtime_error("transformPoint: division by zero");
    }
    
    return cv::Point2d(
        (T.at<double>(0,0) * x + T.at<double>(0,1) * y + T.at<double>(0,2)) / w,
        (T.at<double>(1,0) * x + T.at<double>(1,1) * y + T.at<double>(1,2)) / w
    );
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
