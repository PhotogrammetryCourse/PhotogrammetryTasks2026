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

    cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs_in, const cv::Vec3d *xs_in, int count)
    {
        using mat = Eigen::MatrixXd;

        cv::Vec3d c3(0, 0, 0);
        for (int i = 0; i < count; ++i) c3 += Xs_in[i];
        c3 *= (1.0 / count);
        double rms3 = 0;
        for (int i = 0; i < count; ++i) {
            cv::Vec3d d = Xs_in[i] - c3;
            rms3 += d.dot(d);
        }
        rms3 = std::sqrt(rms3 / (3.0 * count));
        if (rms3 == 0) rms3 = 1;
        double s3 = std::sqrt(3.0) / rms3;
        cv::Matx44d U(s3, 0, 0, -s3 * c3[0],
                      0, s3, 0, -s3 * c3[1],
                      0, 0, s3, -s3 * c3[2],
                      0, 0, 0, 1);
        cv::Matx44d Uinv(1.0 / s3, 0, 0, c3[0],
                         0, 1.0 / s3, 0, c3[1],
                         0, 0, 1.0 / s3, c3[2],
                         0, 0, 0, 1);

        std::vector<cv::Vec3d> xs_n(count);
        cv::Vec2d c2(0, 0);
        for (int i = 0; i < count; ++i) {
            double w = xs_in[i][2];
            c2 += cv::Vec2d(xs_in[i][0] / w, xs_in[i][1] / w);
        }
        c2 *= (1.0 / count);
        double rms2 = 0;
        for (int i = 0; i < count; ++i) {
            double w = xs_in[i][2];
            cv::Vec2d d(xs_in[i][0] / w - c2[0], xs_in[i][1] / w - c2[1]);
            rms2 += d.dot(d);
        }
        rms2 = std::sqrt(rms2 / (2.0 * count));
        if (rms2 == 0) rms2 = 1;
        double s2 = std::sqrt(2.0) / rms2;
        cv::Matx33d T(s2, 0, -s2 * c2[0],
                      0, s2, -s2 * c2[1],
                      0, 0, 1);
        cv::Matx33d Tinv(1.0 / s2, 0, c2[0],
                         0, 1.0 / s2, c2[1],
                         0, 0, 1);

        std::vector<cv::Vec3d> Xs_n(count);
        for (int i = 0; i < count; ++i) {
            cv::Vec4d Xh(Xs_in[i][0], Xs_in[i][1], Xs_in[i][2], 1);
            cv::Vec4d Xn = U * Xh;
            Xs_n[i] = cv::Vec3d(Xn[0], Xn[1], Xn[2]);
            xs_n[i] = T * xs_in[i];
        }

        mat A(2 * count, 12);
        A.setZero();

        for (int i = 0; i < count; ++i) {
            double x = xs_n[i][0], y = xs_n[i][1], w = xs_n[i][2];
            double X = Xs_n[i][0], Y = Xs_n[i][1], Z = Xs_n[i][2];
            double W = 1.0;

            A(2 * i,     4)  = -w * X; A(2 * i,     5)  = -w * Y; A(2 * i,     6)  = -w * Z; A(2 * i,     7)  = -w * W;
            A(2 * i,     8)  =  y * X; A(2 * i,     9)  =  y * Y; A(2 * i,     10) =  y * Z; A(2 * i,     11) =  y * W;
            A(2 * i + 1, 0)  =  w * X; A(2 * i + 1, 1)  =  w * Y; A(2 * i + 1, 2)  =  w * Z; A(2 * i + 1, 3)  =  w * W;
            A(2 * i + 1, 8)  = -x * X; A(2 * i + 1, 9)  = -x * Y; A(2 * i + 1, 10) = -x * Z; A(2 * i + 1, 11) = -x * W;
        }

        Eigen::JacobiSVD<mat> svd(A, Eigen::ComputeFullV);
        Eigen::VectorXd p = svd.matrixV().col(11);

        matrix34d P_n;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                P_n(i, j) = p[i * 4 + j];

        matrix34d P_nU;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int k = 0; k < 4; ++k) s += P_n(i, k) * U(k, j);
                P_nU(i, j) = s;
            }
        }
        matrix34d P_orig;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += Tinv(i, k) * P_nU(k, j);
                P_orig(i, j) = s;
            }
        }

        return canonicalizeP(P_orig);
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

        const double T2 = threshold_px * threshold_px;
        double best_cost = std::numeric_limits<double>::max();
        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            double cost = 0;
            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                cv::Vec3d px_h = calib.project(P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1));
                cv::Vec2d px = {px_h[0] / px_h[2], px_h[1] / px_h[2]};
                cv::Vec2d d = px - x[i];
                double e2 = d.dot(d);
                if (e2 < T2) {
                    cost += e2;
                    ++support;
                } else {
                    cost += T2;
                }
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_support = support;
                best_P = P;

                std::cout << "estimateCameraMatrixRANSAC : MSAC cost: " << best_cost << " support: " << best_support << "/" << n_points << std::endl;
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        return best_P;
    }

}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
