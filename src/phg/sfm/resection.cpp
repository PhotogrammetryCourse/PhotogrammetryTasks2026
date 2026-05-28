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

        // каждая точка даёт 2 уравнения, 12 неизвестных (элементы P 3×4)
        mat A(2 * count, 12);
        A.setZero();

        for (int i = 0; i < count; ++i) {
            double x = xs[i][0];
            double y = xs[i][1];
            double w = xs[i][2];  // = 1 после unproject

            double X = Xs[i][0];
            double Y = Xs[i][1];
            double Z = Xs[i][2];
            double W = 1.0;

            // y*(P^3·Xh) - w*(P^2·Xh) = 0
            A.row(2*i)   << 0,   0,   0,   0,
                            -w*X,-w*Y,-w*Z,-w*W,
                             y*X, y*Y, y*Z, y*W;

            // w*(P^1·Xh) - x*(P^3·Xh) = 0
            A.row(2*i+1) <<  w*X, w*Y, w*Z, w*W,
                             0,   0,   0,   0,
                            -x*X,-x*Y,-x*Z,-x*W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd p = svd.matrixV().col(11);

        matrix34d result;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                result(r, c) = p[4*r + c];

        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();

        // для P 3×4 нужно 11 DOF, каждая точка даёт 2 ур-я → минимум 6 точек
        const int n_samples = 6;
        const int n_trials = 1000;
        const double threshold_px = 3;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d Xs_s[n_samples];  // 3D мировые точки
            cv::Vec3d xs_s[n_samples];  // нормализованные лучи (без K)
            for (int i = 0; i < n_samples; ++i) {
                Xs_s[i] = X[sample[i]];
                xs_s[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(Xs_s, xs_s, n_samples);

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                // проецируем: сначала P (без K), потом K
                cv::Vec3d x_cam = P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1.0);
                cv::Vec3d x_px  = calib.project(x_cam);
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
