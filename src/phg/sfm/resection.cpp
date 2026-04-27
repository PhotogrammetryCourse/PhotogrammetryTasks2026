#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "sfm_utils.h"
#include "defines.h"

namespace {

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
        using mat = Eigen::MatrixXd;

        mat A(2 * count, 12);
        A.setZero();

        for (int i = 0; i < count; ++i) {
            double x = xs[i][0], y = xs[i][1], w = xs[i][2];
            double X = Xs[i][0], Y = Xs[i][1], Z = Xs[i][2];
            double W = 1.0;

            // eq1: y*(P3*X) - w*(P2*X) = 0
            A(2 * i,     4)  = -w * X; A(2 * i,     5)  = -w * Y; A(2 * i,     6)  = -w * Z; A(2 * i,     7)  = -w * W;
            A(2 * i,     8)  =  y * X; A(2 * i,     9)  =  y * Y; A(2 * i,     10) =  y * Z; A(2 * i,     11) =  y * W;
            // eq2: w*(P1*X) - x*(P3*X) = 0
            A(2 * i + 1, 0)  =  w * X; A(2 * i + 1, 1)  =  w * Y; A(2 * i + 1, 2)  =  w * Z; A(2 * i + 1, 3)  =  w * W;
            A(2 * i + 1, 8)  = -x * X; A(2 * i + 1, 9)  = -x * Y; A(2 * i + 1, 10) = -x * Z; A(2 * i + 1, 11) = -x * W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd p = svd.matrixV().col(11);

        matrix34d result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                result(i, j) = p[i * 4 + j];

        return canonicalizeP(result);
    }

    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();
        const int n_trials = 10000;
        const double threshold_px = 8;
        const int n_samples = 6;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples]; // 3D points
            cv::Vec3d ms1[n_samples]; // unprojected rays
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                cv::Vec3d px_h = calib.project(P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1));
                cv::Vec2d px = {px_h[0] / px_h[2], px_h[1] / px_h[2]};
                if (cv::norm(px - x[i]) < threshold_px) {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_P = P;

                std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

                if (best_support == n_points) {
                    break;
                }
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        {
            std::vector<cv::Point3d> obj_pts;
            std::vector<cv::Point2d> img_pts;
            for (int i = 0; i < n_points; ++i) {
                cv::Vec3d px_h = calib.project(best_P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1));
                cv::Vec2d px = {px_h[0] / px_h[2], px_h[1] / px_h[2]};
                if (cv::norm(px - x[i]) < threshold_px) {
                    obj_pts.push_back({X[i][0], X[i][1], X[i][2]});
                    img_pts.push_back({x[i][0], x[i][1]});
                }
            }

            cv::Mat K_mat(calib.K());
            cv::Mat R_init(3, 3, CV_64F);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    R_init.at<double>(r, c) = best_P(r, c);
            cv::Mat rvec, tvec(3, 1, CV_64F);
            cv::Rodrigues(R_init, rvec);
            tvec.at<double>(0) = best_P(0, 3);
            tvec.at<double>(1) = best_P(1, 3);
            tvec.at<double>(2) = best_P(2, 3);

            cv::solvePnP(obj_pts, img_pts, K_mat, cv::noArray(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

            cv::Mat R_refined;
            cv::Rodrigues(rvec, R_refined);
            matrix34d P_lm;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) P_lm(r, c) = R_refined.at<double>(r, c);
                P_lm(r, 3) = tvec.at<double>(r);
            }

            int lm_support = 0;
            for (int i = 0; i < n_points; ++i) {
                cv::Vec3d px_h = calib.project(P_lm * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1));
                cv::Vec2d px = {px_h[0] / px_h[2], px_h[1] / px_h[2]};
                if (cv::norm(px - x[i]) < threshold_px) ++lm_support;
            }
            if (lm_support >= best_support * 0.9) {
                best_P = P_lm;
                std::cout << "estimateCameraMatrixRANSAC : LM support: " << lm_support << "/" << n_points << std::endl;
            }
        }

        return best_P;
    }

}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
