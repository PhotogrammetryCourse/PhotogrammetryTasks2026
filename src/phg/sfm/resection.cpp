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

        for (int i = 0; i < count; ++i) {
            const double x = xs[i][0];
            const double y = xs[i][1];
            const double w = xs[i][2];

            const double X = Xs[i][0];
            const double Y = Xs[i][1];
            const double Z = Xs[i][2];
            const double W = 1.0;

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
        for (int i = 0; i < 12; ++i) {
            result(i / 4, i % 4) = p[i];
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
        for (size_t i = 0; i < X.size(); ++i) {
            object_points.emplace_back(X[i][0], X[i][1], X[i][2]);
            image_points.emplace_back(x[i][0], x[i][1]);
        }

        const cv::Matx33d Kx = calib.K();
        cv::Mat K(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                K.at<double>(r, c) = Kx(r, c);
            }
        }

        cv::Mat rvec, tvec, inliers;
        const bool ok = cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec, false,
                                           1000, 3.0, 0.999, inliers, cv::SOLVEPNP_EPNP);
        if (!ok) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        cv::solvePnP(object_points, image_points, K, cv::noArray(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

        cv::Mat Rm;
        cv::Rodrigues(rvec, Rm);

        matrix34d result;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result(r, c) = Rm.at<double>(r, c);
            }
            result(r, 3) = tvec.at<double>(r, 0);
        }

        return canonicalizeP(result);
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
