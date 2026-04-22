#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <iostream>
#include <Eigen/SVD>
#include <numeric>
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
       int a_rows = count;
       int a_cols = 9;

       Eigen::MatrixXd A(a_rows, a_cols);

       for (int i_pair = 0; i_pair < count; ++i_pair) {

           double x0 = m0[i_pair][0];
           double y0 = m0[i_pair][1];

           double x1 = m1[i_pair][0];
           double y1 = m1[i_pair][1];


            A(i_pair, 0) = x1 * x0;
            A(i_pair, 1) = x1 * y0;
            A(i_pair, 2) = x1;
            A(i_pair, 3) = y1 * x0;
            A(i_pair, 4) = y1 * y0;
            A(i_pair, 5) = y1;
            A(i_pair, 6) = x0;
            A(i_pair, 7) = y0;
            A(i_pair, 8) = 1;
       }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto v_mat = svda.matrixV();
        Eigen::VectorXd f = v_mat.col(v_mat.cols() - 1);

        Eigen::MatrixXd F_tmp(3, 3);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                F_tmp(r, c) = f(r * 3 + c);
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd_final(F_tmp, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::VectorXd s_vals = svd_final.singularValues();
        s_vals(2) = 0.0; // Обнуляем последнюю компоненту

        Eigen::MatrixXd F_final = svd_final.matrixU() * s_vals.asDiagonal() * svd_final.matrixV().transpose();

        cv::Matx33d out;
        copy(F_final, out);

        return out;
    }

    // Нужно создать матрицу преобразования, которая сдвинет переданное множество точек так, что центр масс перейдет в ноль, а Root Mean Square расстояние до него станет sqrt(2)
    // (см. Hartley & Zisserman p.107 Why is normalization essential?)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m)
    {
        const double n = static_cast<double>(m.size());

        cv::Vec2d avg = std::accumulate(m.begin(), m.end(), cv::Vec2d(0, 0));
        avg *= (1.0 / n);

        double deviation_sum = 0.0;
        for (const auto &p : m) {
            const double dx = p[0] - avg[0];
            const double dy = p[1] - avg[1];
            deviation_sum += (dx * dx + dy * dy);
        }

        const double ratio = std::sqrt(2.0) / std::sqrt(deviation_sum / n);

        cv::Matx33d T = cv::Matx33d::eye();
        T(0, 0) = ratio;
        T(1, 1) = ratio;
        T(0, 2) = -ratio * avg[0];
        T(1, 2) = -ratio * avg[1];

        return T;
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d>& m0, const std::vector<cv::Vec2d>& m1, double threshold_px)
    {
        if (m0.size() < 8) {
            throw std::runtime_error("estimateFMatrixRANSAC: too few correspondences");
        }

        const int n_matches = static_cast<int>(m0.size());

        cv::Matx33d TN0 = getNormalizeTransform(m0);
        cv::Matx33d TN1 = getNormalizeTransform(m1);

        std::vector<cv::Vec2d> m0_t(n_matches);
        std::vector<cv::Vec2d> m1_t(n_matches);
        for (int i = 0; i < n_matches; i++) {
            m0_t[i] = transformPoint(m0[i], TN0);
            m1_t[i] = transformPoint(m1[i], TN1);
        }

        if (n_matches == 8) {
            cv::Matx33d F = estimateFMatrixDLT(m0_t.data(), m1_t.data(), 8);
            return TN1.t() * F * TN0;
        }

        const int n_trials = 5000;
        const int n_samples = 8;
        uint64_t seed = 12345;

        int best_support = 0;
        cv::Matx33d best_F_norm = cv::Matx33d::eye();
        std::vector<int> best_inliers;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_matches, n_samples, &seed);

            cv::Vec2d ms0[8], ms1[8];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = m0_t[sample[i]];
                ms1[i] = m1_t[sample[i]];
            }

            cv::Matx33d F = estimateFMatrixDLT(ms0, ms1, 8);

            cv::Matx33d F_denorm = TN1.t() * F * TN0;

            int support = 0;
            std::vector<int> current_inliers;
            for (int i = 0; i < n_matches; ++i) {
                if (phg::epipolarTest(m0[i], m1[i], F_denorm, threshold_px)) {
                    ++support;
                    current_inliers.push_back(i);
                }
            }

            if (support > best_support) {
                best_support = support;
                best_F_norm = F;
                best_inliers = std::move(current_inliers);

                if (best_support > 0.95 * n_matches) {
                    break;
                }
            }
        }

        if (best_support < 8) {
            throw std::runtime_error("estimateFMatrixRANSAC : failed to find valid model");
        }


        std::vector<cv::Vec2d> in0(best_support), in1(best_support);
        for (int i = 0; i < best_support; ++i) {
            in0[i] = m0_t[best_inliers[i]];
            in1[i] = m1_t[best_inliers[i]];
        }

        cv::Matx33d F_final_norm = estimateFMatrixDLT(in0.data(), in1.data(), best_support);

        Eigen::Matrix3d Fe;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Fe(i, j) = F_final_norm(i, j);
            }
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(Fe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d s = svd.singularValues();
        s[2] = 0; // Force rank 2
        Fe = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                F_final_norm(i, j) = Fe(i, j);
            }
        }

        return TN1.t() * F_final_norm * TN0;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector <cv::Vec2d> &m0, const std::vector <cv::Vec2d> &m1, double threshold_px) {
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
