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
            // fill 2 rows of matrix A

            double x0 = xs0[i];
            double y0 = ys0[i];
            double w0 = ws0[i];

            double x1 = xs1[i];
            double y1 = ys1[i];
            double w1 = ws1[i];

            // 8 elements of matrix + free term as needed by gauss routine
            A.push_back({x0*w1, y0*w1, w0*w1, 0, 0, 0, -x1*x0, -x1*y0, x1*w0});
            A.push_back({0, 0, 0, x0*w1, y0*w1, w0*w1, -y1*x0, -y1*y0, y1*w0});
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
            for (double i : xs0) {
                std::cerr << i << ", ";
            }
            std::cerr << "\ngauss: ys0: ";
            for (double i : ys0) {
                std::cerr << i << ", ";
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

        // Дополнительный балл, если вместо обычной версии будет использована модификация a-contrario RANSAC (реализуем как раз ее!)
        // * [1] Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers. (Lionel Moisan, Pierre Moulon, Pascal Monasse)
        // * [2] Adaptive Structure from Motion with a contrario model estimation. (Pierre Moulon, Pascal Monasse, Renaud Marlet)
        // * (простое описание для понимания)
        // * [3] http://ikrisoft.blogspot.com/2015/01/ransac-with-contrario-approach.html
        //
        // Идея: вместо фиксированного порога ошибки репроекции используем NFA (Number of False Alarms).
        // NFA(k) = C(n,p) * C(n-p, k-p) * alpha(r_k)^(k-p)
        // где p=4 (минимальная выборка для гомографии),
        // alpha(r) = pi * r^2 / A — вероятность попадания случайной точки в круг радиуса r (H0: равномерное распределение),
        // r_k — k-я по величине невязка (отсортированная).
        // Модель принимается, если NFA < 1 (log(NFA) < 0).
        // Оптимальный порог k* выбирается как argmin_k NFA(k) для каждой гипотезы.

        const int n_matches = static_cast<int>(points_lhs.size());

        const int n_trials = 1000;
        uint64_t seed = 1;

        const int n_samples = 4;

        // Оценка площади области целевого изображения для нулевой гипотезы H0
        // (при H0 точки равномерно распределены в этой области)
        float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();
        for (const auto &pt : points_rhs) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }
        double area = static_cast<double>(max_x - min_x) * (max_y - min_y);
        area = std::max(area, 1.0);

        // log C(n, p) — число возможных минимальных подвыборок
        double log_c_n_p = std::lgamma(n_matches + 1) - std::lgamma(n_samples + 1) - std::lgamma(n_matches - n_samples + 1);

        double best_log_nfa = 0.0;
        cv::Mat best_H;
        int best_support = 0;

        std::vector<int> sample;
        std::vector<double> residuals(n_matches);
        std::vector<double> sorted_residuals(n_matches);

        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            randomSample(sample, n_matches, n_samples, &seed);

            cv::Mat H;
            try {
                H = estimateHomography4Points(points_lhs[sample[0]], points_lhs[sample[1]], points_lhs[sample[2]], points_lhs[sample[3]],
                                              points_rhs[sample[0]], points_rhs[sample[1]], points_rhs[sample[2]], points_rhs[sample[3]]);
            } catch (const std::exception &e) {
                continue;
            }

            for (int i = 0; i < n_matches; ++i) {
                try {
                    cv::Point2d proj = phg::transformPoint(points_lhs[i], H);
                    residuals[i] = cv::norm(proj - cv::Point2d(points_rhs[i]));
                } catch (const std::exception &e) {
                    residuals[i] = 1e10;
                    std::cerr << e.what() << std::endl;
                }
            }

            sorted_residuals = residuals;
            std::sort(sorted_residuals.begin(), sorted_residuals.end());

            // Для каждого k (возможное число инлаеров) вычисляем NFA в лог-пространстве:
            // log(NFA(k)) = log C(n,p) + log C(n-p, k-p) + (k-p) * log(pi * r_k^2 / A)
            // Ищем k* = argmin_k log(NFA(k))
            double trial_best_log_nfa = std::numeric_limits<double>::max();
            int trial_best_k = 0;

            int m = n_matches - n_samples;

            for (int k = n_samples + 1; k <= n_matches; ++k) {
                double r_k = sorted_residuals[k - 1];
                double alpha = CV_PI * r_k * r_k / area;
                if (alpha >= 1.0) break; // alpha >= 1 => NFA растёт дальше
                if (alpha <= 0.0) continue;

                int j = k - n_samples;
                double log_c_m_j = std::lgamma(m + 1) - std::lgamma(j + 1) - std::lgamma(m - j + 1);
                double log_nfa = log_c_n_p + log_c_m_j + j * std::log(alpha);

                if (log_nfa < trial_best_log_nfa) {
                    trial_best_log_nfa = log_nfa;
                    trial_best_k = k;
                }
            }

            if (trial_best_log_nfa < best_log_nfa) {
                best_log_nfa = trial_best_log_nfa;
                best_H = H;
                best_support = trial_best_k;

                std::cout << "AC-RANSAC: log(NFA)=" << best_log_nfa
                          << ", support=" << best_support << "/" << n_matches << std::endl;

                if (best_support == n_matches) break;
            }
        }

        std::cout << "AC-RANSAC: best log(NFA)=" << best_log_nfa
                  << ", support=" << best_support << "/" << n_matches << std::endl;

        if (best_H.empty()) {
            throw std::runtime_error("AC-RANSAC: no meaningful model found (log(NFA) >= 0)");
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
    cv::Mat T64;
    T.convertTo(T64, CV_64FC1);
    const double *t = T64.ptr<double>();

    double x = pt.x;
    double y = pt.y;

    double wx = t[0] * x + t[1] * y + t[2];
    double wy = t[3] * x + t[4] * y + t[5];
    double w  = t[6] * x + t[7] * y + t[8];

    if (std::abs(w) < 1e-12) {
        throw std::runtime_error("transformPoint: w is near zero");
    }

    return {wx / w, wy / w};
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
