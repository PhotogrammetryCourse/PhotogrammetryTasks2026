#include "homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

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
            double x0 = xs0[i];
            double y0 = ys0[i];
            double w0 = ws0[i];

            double x1 = xs1[i];
            double y1 = ys1[i];
            double w1 = ws1[i];

            A.push_back({ x0*w1,  y0*w1,  w0*w1,    0,     0,     0,   -x0*x1, -y0*x1,   w0*x1 });
            A.push_back({   0,     0,     0,    x0*w1, y0*w1, w0*w1,   -x0*y1, -y0*y1,   w0*y1 });
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

    cv::Mat estimateHomographyNPoints(const std::vector<cv::Point2f> &pts_lhs, const std::vector<cv::Point2f> &pts_rhs)
    {
        const int n = pts_lhs.size();
        cv::Mat A(2 * n, 9, CV_64FC1, 0.0);
        for (int i = 0; i < n; ++i) {
            double x0 = pts_lhs[i].x, y0 = pts_lhs[i].y;
            double x1 = pts_rhs[i].x, y1 = pts_rhs[i].y;
            double *r0 = A.ptr<double>(2 * i);
            double *r1 = A.ptr<double>(2 * i + 1);
            r0[0]=x0; r0[1]=y0; r0[2]=1; r0[3]=0; r0[4]=0; r0[5]=0; r0[6]=-x0*x1; r0[7]=-y0*x1; r0[8]=-x1;
            r1[0]=0;  r1[1]=0;  r1[2]=0; r1[3]=x0; r1[4]=y0; r1[5]=1; r1[6]=-x0*y1; r1[7]=-y0*y1; r1[8]=-y1;
        }
        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt);
        cv::Mat h = vt.row(8).reshape(1, 3);
        h.convertTo(h, CV_64FC1);
        return h / h.at<double>(2, 2);
    }

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
        }


        const int n_matches = points_lhs.size();
        const int n_trials = 500;
        const int n_samples = 4;
        uint64_t seed = 1;

        double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;
        for (const auto &p : points_rhs) {
            if (p.x < min_x) min_x = p.x; if (p.x > max_x) max_x = p.x;
            if (p.y < min_y) min_y = p.y; if (p.y > max_y) max_y = p.y;
        }
        double area = std::max(1.0, (max_x - min_x) * (max_y - min_y));

        const double log_n_tests = std::log((double)n_trials * n_matches);

        auto logNFA = [&](const std::vector<double> &sorted_res, int k) -> double {
            double eps = sorted_res[k - 1];
            double p = M_PI * eps * eps / area;
            if (p <= 0.0 || p >= 1.0) return 1e18;
            double log_cnk = std::lgamma(n_matches + 1) - std::lgamma(k + 1) - std::lgamma(n_matches - k + 1);
            return log_n_tests + log_cnk + k * std::log(p) + (n_matches - k) * std::log(1.0 - p);
        };

        double best_log_nfa = 0.0;
        double best_threshold = 2.0;
        cv::Mat best_H;
        int best_k = 0;

        std::vector<int> sample;
        std::vector<double> residuals(n_matches);

        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            randomSample(sample, n_matches, n_samples, &seed);

            cv::Mat H;
            try {
                H = estimateHomography4Points(
                    points_lhs[sample[0]], points_lhs[sample[1]], points_lhs[sample[2]], points_lhs[sample[3]],
                    points_rhs[sample[0]], points_rhs[sample[1]], points_rhs[sample[2]], points_rhs[sample[3]]);
            } catch (...) { continue; }

            bool valid = true;
            for (int i = 0; i < n_matches; ++i) {
                try {
                    cv::Point2d proj = phg::transformPoint(points_lhs[i], H);
                    residuals[i] = cv::norm(proj - cv::Point2d(points_rhs[i]));
                } catch (...) { valid = false; break; }
            }
            if (!valid) continue;

            std::vector<double> sorted_res = residuals;
            std::sort(sorted_res.begin(), sorted_res.end());

            for (int k = 4; k <= n_matches; ++k) {
                double lnfa = logNFA(sorted_res, k);
                if (lnfa < best_log_nfa) {
                    best_log_nfa = lnfa;
                    best_threshold = sorted_res[k - 1];
                    best_H = H;
                    best_k = k;
                }
            }
        }

        if (best_H.empty()) {
            throw std::runtime_error("estimateHomographyRANSAC : failed to find meaningful homography");
        }

        std::cout << "estimateHomographyRANSAC : best log_nfa=" << best_log_nfa
                  << ", threshold=" << best_threshold << ", inliers=" << best_k << "/" << n_matches << std::endl;

        std::vector<cv::Point2f> inlier_lhs, inlier_rhs;
        for (int i = 0; i < n_matches; ++i) {
            try {
                cv::Point2d proj = phg::transformPoint(points_lhs[i], best_H);
                if (cv::norm(proj - cv::Point2d(points_rhs[i])) <= best_threshold) {
                    inlier_lhs.push_back(points_lhs[i]);
                    inlier_rhs.push_back(points_rhs[i]);
                }
            } catch (...) {}
        }
        if ((int)inlier_lhs.size() >= 4) {
            best_H = estimateHomographyNPoints(inlier_lhs, inlier_rhs);
        }

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
    const double* h = T.ptr<double>();
    double x = h[0] * pt.x + h[1] * pt.y + h[2];
    double y = h[3] * pt.x + h[4] * pt.y + h[5];
    double w = h[6] * pt.x + h[7] * pt.y + h[8];
    if (std::abs(w) < 1e-10)
        throw std::runtime_error("transformPoint: w is zero");
    return { x / w, y / w };
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
