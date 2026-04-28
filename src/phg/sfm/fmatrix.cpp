#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <iostream>
#include <cmath>
#include <Eigen/SVD>
#include <opencv2/calib3d.hpp>

namespace {

    void infoF(const cv::Matx33d &Fcv)
    {
        Eigen::MatrixXd F;
        copy(Fcv, F);

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd U = svdf.matrixU();
        Eigen::VectorXd s = svdf.singularValues();
        Eigen::MatrixXd V = svdf.matrixV();

        std::cout << "F info:\nF:\n" << F << "\nU:\n" << U << "\ns:\n" << s << "\nV:\n" << V << std::endl;
    }

    // (см. Hartley & Zisserman p.279)
    cv::Matx33d estimateFMatrixDLT(const cv::Vec2d *m0, const cv::Vec2d *m1, int count)
    {
        Eigen::MatrixXd A(count, 9);

        for (int i = 0; i < count; ++i) {
            double x0 = m0[i][0], y0 = m0[i][1];
            double x1 = m1[i][0], y1 = m1[i][1];
            A.row(i) << x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, 1;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd f = svda.matrixV().col(8);

        Eigen::MatrixXd F(3, 3);
        F.row(0) << f[0], f[1], f[2];
        F.row(1) << f[3], f[4], f[5];
        F.row(2) << f[6], f[7], f[8];

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svdf.singularValues();
        s[2] = 0;
        F = svdf.matrixU() * s.asDiagonal() * svdf.matrixV().transpose();

        cv::Matx33d Fcv;
        copy(F, Fcv);
        return Fcv;
    }

    // (см. Hartley & Zisserman p.107 Why is normalization essential?)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m)
    {
        double cx = 0, cy = 0;
        for (const auto &p : m) {
            cx += p[0];
            cy += p[1];
        }
        cx /= m.size();
        cy /= m.size();

        double rms = 0;
        for (const auto &p : m) {
            double dx = p[0] - cx, dy = p[1] - cy;
            rms += dx * dx + dy * dy;
        }
        rms = std::sqrt(rms / m.size());

        double scale = std::sqrt(2.0) / rms;

        return cv::Matx33d(scale, 0, -scale * cx,
                           0, scale, -scale * cy,
                           0, 0, 1);
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixRANSAC: m0.size() != m1.size()");
        }

        const int n_matches = m0.size();

        cv::Matx33d TN0 = getNormalizeTransform(m0);
        cv::Matx33d TN1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0_t(n_matches);
        std::vector<cv::Vec2d> m1_t(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            m0_t[i] = transformPoint(m0[i], TN0);
            m1_t[i] = transformPoint(m1[i], TN1);
        }

        {
            cv::Matx33d TN0_check = getNormalizeTransform(m0_t);
            cv::Matx33d TN1_check = getNormalizeTransform(m1_t);
            (void)TN0_check;
            (void)TN1_check;
        }

        const int n_samples = 8;
        const int n_trials = (n_matches > 10000) ? 10000 : 1000;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx33d best_F;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_matches, n_samples, &seed);

            cv::Vec2d ms0[n_samples];
            cv::Vec2d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = m0_t[sample[i]];
                ms1[i] = m1_t[sample[i]];
            }

            cv::Matx33d Fn = estimateFMatrixDLT(ms0, ms1, n_samples);

            cv::Matx33d F = TN1.t() * Fn * TN0;

            int support = 0;
            for (int i = 0; i < n_matches; ++i) {
                if (phg::epipolarTest(m0[i], m1[i], F, threshold_px) && phg::epipolarTest(m1[i], m0[i], F.t(), threshold_px)) {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_F = F;

                std::cout << "estimateFMatrixRANSAC : support: " << best_support << "/" << n_matches << std::endl;
                infoF(F);

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        std::cout << "estimateFMatrixRANSAC : best support: " << best_support << "/" << n_matches << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateFMatrixRANSAC : failed to estimate fundamental matrix");
        }

        return best_F;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px) {
    return estimateFMatrixRANSAC(m0, m1, threshold_px);
}

cv::Matx33d phg::findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px) {
    return cv::findFundamentalMat(m0, m1, cv::FM_RANSAC, threshold_px);
}

cv::Matx33d phg::composeFMatrix(const cv::Matx34d &P0, const cv::Matx34d &P1)
{
    // compute fundamental matrix from general cameras
    // Hartley & Zisserman (17.3 - p412)

    cv::Matx33d F;

#define det4(a, b, c, d) \
      ((a)(0) * (b)(1) - (a)(1) * (b)(0)) * ((c)(2) * (d)(3) - (c)(3) * (d)(2)) - \
      ((a)(0) * (b)(2) - (a)(2) * (b)(0)) * ((c)(1) * (d)(3) - (c)(3) * (d)(1)) + \
      ((a)(0) * (b)(3) - (a)(3) * (b)(0)) * ((c)(1) * (d)(2) - (c)(2) * (d)(1)) + \
      ((a)(1) * (b)(2) - (a)(2) * (b)(1)) * ((c)(0) * (d)(3) - (c)(3) * (d)(0)) - \
      ((a)(1) * (b)(3) - (a)(3) * (b)(1)) * ((c)(0) * (d)(2) - (c)(2) * (d)(0)) + \
      ((a)(2) * (b)(3) - (a)(3) * (b)(2)) * ((c)(0) * (d)(1) - (c)(1) * (d)(0))

    int i, j;
    for (j = 0; j < 3; j++)
        for (i = 0; i < 3; i++) {
            // here the sign is encoded in the order of lines ~ai
            const auto a1 = P0.row((i + 1) % 3);
            const auto a2 = P0.row((i + 2) % 3);
            const auto b1 = P1.row((j + 1) % 3);
            const auto b2 = P1.row((j + 2) % 3);

            F(j, i) = det4(a1, a2, b1, b2);
        }

#undef det4

    return F;
}
