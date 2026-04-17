#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
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
        using vec = Eigen::VectorXd;

        mat A(2 * count, 12);

        for (int i = 0; i < count; ++i) {

            double x = xs[i][0];
            double y = xs[i][1];
            double w = xs[i][2];

            double X = Xs[i][0];
            double Y = Xs[i][1];
            double Z = Xs[i][2];
            double W = 1.0;

            A(2 * i, 0) = 0;     A(2 * i, 1) = 0;     A(2 * i, 2) = 0;     A(2 * i, 3) = 0;
            A(2 * i, 4) = -w*X;  A(2 * i, 5) = -w*Y;  A(2 * i, 6) = -w*Z;  A(2 * i, 7) = -w*W;
            A(2 * i, 8) = y*X;   A(2 * i, 9) = y*Y;   A(2 * i, 10) = y*Z;  A(2 * i, 11) = y*W;

            A(2 * i + 1, 0) = w*X;   A(2 * i + 1, 1) = w*Y;   A(2 * i + 1, 2) = w*Z;   A(2 * i + 1, 3) = w*W;
            A(2 * i + 1, 4) = 0;     A(2 * i + 1, 5) = 0;     A(2 * i + 1, 6) = 0;     A(2 * i + 1, 7) = 0;
            A(2 * i + 1, 8) = -x*X;  A(2 * i + 1, 9) = -x*Y;  A(2 * i + 1, 10) = -x*Z; A(2 * i + 1, 11) = -x*W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        vec p = svd.matrixV().col(11);

        matrix34d result;
        for (int i = 0; i < 12; ++i) {
            result(i / 4, i % 4) = p(i);
        }

        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = 2000;

        const double threshold_px = 3;

        const int n_samples = 6;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[6];
            cv::Vec3d ms1[6];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                vector3d x_cam;
                x_cam[0] = P(0, 0) * X[i][0] + P(0, 1) * X[i][1] + P(0, 2) * X[i][2] + P(0, 3);
                x_cam[1] = P(1, 0) * X[i][0] + P(1, 1) * X[i][1] + P(1, 2) * X[i][2] + P(1, 3);
                x_cam[2] = P(2, 0) * X[i][0] + P(2, 1) * X[i][1] + P(2, 2) * X[i][2] + P(2, 3);
                vector3d x_px = calib.project(x_cam);
                cv::Vec2d px(x_px[0] / x_px[2], x_px[1] / x_px[2]);
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

        // после падения тестов на маке, попробуем добавить рефайн и сюда
        auto countSupport = [&](const cv::Matx34d &P, std::vector<int> *inlier_ids) -> int {
            int s = 0;
            if (inlier_ids) inlier_ids->clear();
            for (int i = 0; i < n_points; ++i) {
                vector3d x_cam;
                x_cam[0] = P(0, 0) * X[i][0] + P(0, 1) * X[i][1] + P(0, 2) * X[i][2] + P(0, 3);
                x_cam[1] = P(1, 0) * X[i][0] + P(1, 1) * X[i][1] + P(1, 2) * X[i][2] + P(1, 3);
                x_cam[2] = P(2, 0) * X[i][0] + P(2, 1) * X[i][1] + P(2, 2) * X[i][2] + P(2, 3);
                vector3d x_px = calib.project(x_cam);
                if (x_px[2] == 0) continue;
                if (cv::Vec2d px(x_px[0] / x_px[2], x_px[1] / x_px[2]); cv::norm(px - x[i]) < threshold_px) {
                    ++s;
                    if (inlier_ids) inlier_ids->push_back(i);
                }
            }
            return s;
        };

        for (int refine_iter = 0; refine_iter < 10; ++refine_iter) {
            std::vector<int> inlier_ids;
            countSupport(best_P, &inlier_ids);

            if (static_cast<int>(inlier_ids.size()) <= n_samples) break;

            std::vector<cv::Vec3d> Xs_in(inlier_ids.size());
            std::vector<cv::Vec3d> xs_in(inlier_ids.size());
            for (size_t i = 0; i < inlier_ids.size(); ++i) {
                Xs_in[i] = X[inlier_ids[i]];
                xs_in[i] = calib.unproject(x[inlier_ids[i]]);
            }

            cv::Matx34d P_refit = estimateCameraMatrixDLT(Xs_in.data(), xs_in.data(), static_cast<int>(inlier_ids.size()));

            int new_support = countSupport(P_refit, nullptr);
            if (new_support <= best_support) break;

            best_support = new_support;
            best_P = P_refit;

            std::cout << "estimateCameraMatrixRANSAC refine: support: " << best_support << "/" << n_points << std::endl;
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        return best_P;
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
