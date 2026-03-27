#include "resection.h"

#include <algorithm>
#include <cmath>
#include <Eigen/SVD>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "sfm_utils.h"
#include "defines.h"

namespace {

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

    // Сделать из первого минора 3х3 матрицу вращения, скомпенсировать масштаб у компоненты сдвига
    matrix34d canonicalizeP(const matrix34d &P)
    {
        matrix3d RR = P.get_minor<3, 3>(0, 0);
        vector3d tt;
        tt[0] = P(0, 3);
        tt[1] = P(1, 3);
        tt[2] = P(2, 3);

        if (cv::determinant(RR) < 0) {
            RR *= -1;
            tt *= -1;
        }

        double sc = 0;
        for (int i = 0; i < 9; i++) {
            sc += RR.val[i] * RR.val[i];
        }
        sc = std::sqrt(3 / sc);

        Eigen::MatrixXd RRe;
        copy(RR, RRe);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RRe = svd.matrixU() * svd.matrixV().transpose();
        copy(RRe, RR);

        tt *= sc;

        matrix34d result;
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = RR(i / 3, i % 3);
        }
        result(0, 3) = tt(0);
        result(1, 3) = tt(1);
        result(2, 3) = tt(2);

        return result;
    }

    // (см. Hartley & Zisserman p.178)
    cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs, const cv::Vec3d *xs, int count)
    {
        if (count < 6) {
            throw std::runtime_error("estimateCameraMatrixDLT: too few correspondences");
        }

        using mat = Eigen::MatrixXd;

        mat A(2 * count, 12);

        for (int i = 0; i < count; ++i) {

            double x = xs[i][0];
            double y = xs[i][1];
            double w = xs[i][2];

            double X = Xs[i][0];
            double Y = Xs[i][1];
            double Z = Xs[i][2];
            double W = 1.0;

            A.row(2 * i + 0) << 0.0, 0.0, 0.0, 0.0,
                                 -w * X, -w * Y, -w * Z, -w * W,
                                  y * X,  y * Y,  y * Z,  y * W;

            A.row(2 * i + 1) << w * X, w * Y, w * Z, w * W,
                                 0.0, 0.0, 0.0, 0.0,
                                -x * X, -x * Y, -x * Z, -x * W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        const Eigen::VectorXd p = svd.matrixV().col(11);

        matrix34d result;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                result(r, c) = p[r * 4 + c];
            }
        }

        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }
        if (X.size() < 6) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: too few correspondences");
        }

        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        object_points.reserve(X.size());
        image_points.reserve(x.size());
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            object_points.emplace_back(X[i][0], X[i][1], X[i][2]);
            image_points.emplace_back(x[i][0], x[i][1]);
        }

        cv::Mat rvec;
        cv::Mat tvec;
        cv::Mat inliers;
        const bool ok = cv::solvePnPRansac(
            object_points,
            image_points,
            calib.K(),
            cv::noArray(),
            rvec,
            tvec,
            false,
            5000,
            3.0,
            0.999,
            inliers,
            cv::SOLVEPNP_EPNP
        );
        if (ok) {
            if (!inliers.empty()) {
                std::vector<cv::Point3d> inlier_object_points;
                std::vector<cv::Point2d> inlier_image_points;
                inlier_object_points.reserve(inliers.rows);
                inlier_image_points.reserve(inliers.rows);
                for (int i = 0; i < inliers.rows; ++i) {
                    const int idx = inliers.at<int>(i, 0);
                    inlier_object_points.push_back(object_points[idx]);
                    inlier_image_points.push_back(image_points[idx]);
                }
                cv::solvePnPRefineLM(inlier_object_points, inlier_image_points, calib.K(), cv::noArray(), rvec, tvec);
            }

            cv::Mat Rmat;
            cv::Rodrigues(rvec, Rmat);

            matrix34d P;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    P(r, c) = Rmat.at<double>(r, c);
                }
                P(r, 3) = tvec.at<double>(r, 0);
            }
            return P;
        }

        const int n_points = X.size();
        const int max_trials = 5000;
        const double threshold_px = 3.0;
        const int n_samples = 6;
        int required_trials = max_trials;
        uint64_t seed = 1;

        std::vector<cv::Vec3d> x_norm(n_points);
        for (int i = 0; i < n_points; ++i) {
            x_norm[i] = calib.unproject(x[i]);
        }

        int best_support = 0;
        cv::Matx34d best_P;
        std::vector<int> best_inliers;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < required_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = x_norm[sample[i]];
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            int support = 0;
            std::vector<int> inliers;
            inliers.reserve(n_points);
            for (int i = 0; i < n_points; ++i) {
                vector4d Xh = {X[i][0], X[i][1], X[i][2], 1.0};
                vector3d cam = P * Xh;
                if (std::abs(cam[2]) <= 1e-12) {
                    continue;
                }

                const vector3d px_h = calib.project(cam);
                cv::Vec2d px = {px_h[0] / px_h[2], px_h[1] / px_h[2]};
                if (cv::norm(px - x[i]) < threshold_px) {
                    ++support;
                    inliers.push_back(i);
                }
            }

            if (support > best_support) {
                best_support = support;
                best_P = P;
                best_inliers = std::move(inliers);

                required_trials = updateRequiredTrials(required_trials, n_points, best_support, n_samples);
                std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

                if (best_support == n_points) {
                    break;
                }
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support < n_samples) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        if (best_inliers.size() >= n_samples) {
            std::vector<cv::Vec3d> inlier_X(best_inliers.size());
            std::vector<cv::Vec3d> inlier_x(best_inliers.size());
            for (int i = 0; i < static_cast<int>(best_inliers.size()); ++i) {
                inlier_X[i] = X[best_inliers[i]];
                inlier_x[i] = x_norm[best_inliers[i]];
            }
            best_P = estimateCameraMatrixDLT(inlier_X.data(), inlier_x.data(), static_cast<int>(best_inliers.size()));
        }

        return best_P;
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
