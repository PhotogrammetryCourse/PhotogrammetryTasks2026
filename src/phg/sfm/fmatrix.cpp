#include "fmatrix.h"
#include "sfm_utils.h"
#include "defines.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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
        if (count < 8) {
            throw std::runtime_error("estimateFMatrixDLT: too few correspondences");
        }

        const int a_rows = count;
        const int a_cols = 9;

        Eigen::MatrixXd A(a_rows, a_cols);

        for (int i_pair = 0; i_pair < count; ++i_pair) {
            double x0 = m0[i_pair][0];
            double y0 = m0[i_pair][1];

            double x1 = m1[i_pair][0];
            double y1 = m1[i_pair][1];

            A.row(i_pair) << x1 * x0, x1 * y0, x1,
                             y1 * x0, y1 * y0, y1,
                             x0,      y0,      1.0;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullV);
        const Eigen::VectorXd null_space = svda.matrixV().col(a_cols - 1);

        Eigen::MatrixXd F(3, 3);
        F.row(0) << null_space[0], null_space[1], null_space[2];
        F.row(1) << null_space[3], null_space[4], null_space[5];
        F.row(2) << null_space[6], null_space[7], null_space[8];

        Eigen::JacobiSVD<Eigen::MatrixXd> svdf(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d s = svdf.singularValues();
        s[2] = 0.0;
        F = svdf.matrixU() * s.asDiagonal() * svdf.matrixV().transpose();

        cv::Matx33d Fcv;
        copy(F, Fcv);

        return Fcv;
    }

    // Нужно создать матрицу преобразования, которая сдвинет переданное множество точек так, что
    // центр масс перейдет в ноль, а Root Mean Square расстояние до него станет sqrt(2).
    // (см. Hartley & Zisserman p.107 Why is normalization essential?)
    cv::Matx33d getNormalizeTransform(const std::vector<cv::Vec2d> &m, bool verbose)
    {
        if (m.empty()) {
            throw std::runtime_error("getNormalizeTransform: empty input");
        }

        cv::Vec2d center(0.0, 0.0);
        for (const cv::Vec2d &pt : m) {
            center += pt;
        }
        center *= 1.0 / double(m.size());

        double mean_dist2 = 0.0;
        for (const cv::Vec2d &pt : m) {
            const cv::Vec2d d = pt - center;
            mean_dist2 += d.dot(d);
        }
        mean_dist2 /= double(m.size());

        const double scale = mean_dist2 > 0.0 ? std::sqrt(2.0 / mean_dist2) : 1.0;
        if (verbose) {
            std::cout << "NORMALIZE TRANSFORM: centroid = " << center << ", scale = " << scale << std::endl;
        }

        return cv::Matx33d(scale, 0.0, -scale * center[0],
                           0.0, scale, -scale * center[1],
                           0.0, 0.0, 1.0);
    }

    cv::Vec2d transformPoint(const cv::Vec2d &pt, const cv::Matx33d &T)
    {
        cv::Vec3d tmp = T * cv::Vec3d(pt[0], pt[1], 1.0);

        if (tmp[2] == 0) {
            throw std::runtime_error("infinite point");
        }

        return cv::Vec2d(tmp[0] / tmp[2], tmp[1] / tmp[2]);
    }

    cv::Matx33d estimateFMatrixRANSAC(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px, bool verbose)
    {
        if (m0.size() != m1.size()) {
            throw std::runtime_error("estimateFMatrixRANSAC: m0.size() != m1.size()");
        }
        if (m0.size() < 8) {
            throw std::runtime_error("estimateFMatrixRANSAC: too few correspondences");
        }

        const int n_matches = static_cast<int>(m0.size());

        if (n_matches == 8) {
            cv::Matx33d TN0 = getNormalizeTransform(m0, verbose);
            cv::Matx33d TN1 = getNormalizeTransform(m1, verbose);

            std::vector<cv::Vec2d> m0_t(n_matches);
            std::vector<cv::Vec2d> m1_t(n_matches);
            for (int i = 0; i < n_matches; ++i) {
                m0_t[i] = transformPoint(m0[i], TN0);
                m1_t[i] = transformPoint(m1[i], TN1);
            }

            return TN1.t() * estimateFMatrixDLT(m0_t.data(), m1_t.data(), n_matches) * TN0;
        }

        cv::Mat inlier_mask;
        cv::Mat Fmat = cv::findFundamentalMat(m0, m1, cv::USAC_MAGSAC, threshold_px, 0.999, inlier_mask);
        if (Fmat.empty()) {
            throw std::runtime_error("estimateFMatrixRANSAC : cv::findFundamentalMat failed");
        }

        cv::Mat F64;
        Fmat.convertTo(F64, CV_64F);

        cv::Matx33d Fcv;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Fcv(r, c) = F64.at<double>(r, c);
            }
        }

        if (verbose) {
            infoF(Fcv);
        }

        return Fcv;
    }

}

cv::Matx33d phg::findFMatrix(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px, bool verbose)
{
    return estimateFMatrixRANSAC(m0, m1, threshold_px, verbose);
}

cv::Matx33d phg::findFMatrixCV(const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, double threshold_px)
{
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
    for (j = 0; j < 3; j++) {
        for (i = 0; i < 3; i++) {
            // here the sign is encoded in the order of lines ~ai
            const auto a1 = P0.row((i + 1) % 3);
            const auto a2 = P0.row((i + 2) % 3);
            const auto b1 = P1.row((j + 1) % 3);
            const auto b2 = P1.row((j + 2) % 3);

            F(j, i) = det4(a1, a2, b1, b2);
        }
    }

#undef det4

    return F;
}
