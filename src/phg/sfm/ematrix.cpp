#include "ematrix.h"

#include "defines.h"
#include "fmatrix.h"
#include "triangulation.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/calib3d.hpp>

namespace {

    // essential matrix must have exactly two equal non zero singular values
    // (см. Hartley & Zisserman p.257)
    void ensureSpectralProperty(matrix3d &Ecv)
    {
        Eigen::Matrix3d matE;
        copy(Ecv, matE);

        Eigen::JacobiSVD<Eigen::Matrix3d> solver(matE, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3d u = solver.matrixU();
        Eigen::Matrix3d v = solver.matrixV();
        Eigen::Vector3d sigma = solver.singularValues();

        if (u.determinant() < 0) {
            u = -u;
        }
        if (v.determinant() < 0) {
            v = -v;
        }

        const double eps = 1e-10;
        if (sigma[0] < eps || sigma[1] < eps) {
            throw std::runtime_error("Essential matrix refinement failed: degenerate singular values");
        }

        double mean_s = (sigma[0] + sigma[1]) * 0.5;

        Eigen::DiagonalMatrix<double, 3> diagS(mean_s, mean_s, 0.0);

        matE = u * diagS * v.transpose();

        copy(matE, Ecv);
    }

}

cv::Matx33d phg::fmatrix2ematrix(const cv::Matx33d &F, const phg::Calibration &calib0, const phg::Calibration &calib1)
{
    const cv::Matx33d k0 = calib0.K();
    const cv::Matx33d k1_t = calib1.K().t();

    cv::Matx33d E = k1_t * F * k0;
    ensureSpectralProperty(E);
    return E;
}

namespace {

    matrix34d composeP(const Eigen::MatrixXd &R, const Eigen::VectorXd &t)
    {
        matrix34d result;

        result(0, 0) = R(0, 0);
        result(0, 1) = R(0, 1);
        result(0, 2) = R(0, 2);
        result(1, 0) = R(1, 0);
        result(1, 1) = R(1, 1);
        result(1, 2) = R(1, 2);
        result(2, 0) = R(2, 0);
        result(2, 1) = R(2, 1);
        result(2, 2) = R(2, 2);

        result(0, 3) = t[0];
        result(1, 3) = t[1];
        result(2, 3) = t[2];

        return result;
    }

    bool depthTest(const vector2d &m0, const vector2d &m1, const phg::Calibration &calib0, const phg::Calibration &calib1, const matrix34d &P0, const matrix34d &P1)
    {
        // скомпенсировать калибровки камер
        vector3d p0 = calib0.unproject(m0);
        vector3d p1 = calib1.unproject(m1);

        vector3d ps[2] = {p0, p1};
        matrix34d Ps[2] = {P0, P1};

        vector4d X = phg::triangulatePoint(Ps, ps, 2);
        if (X[3] != 0) {
            X /= X[3];
        }

        vector3d wrld(X[0], X[1], X[2]);
        const double k_depth0 = calib0.project(wrld)[2];
        const double k_depth1 = calib1.project(wrld)[2];

        // точка должна иметь положительную глубину для обеих камер
        return k_depth0 > 0 && k_depth1 > 0;
    }
}

// Матрицы камер для фундаментальной матрицы определены с точностью до проективного преобразования
// То есть, можно исказить трехмерный мир (применив 4-мерную однородную матрицу), и одновременно поменять матрицы P0, P1 так, что проекции в пикселях не изменятся
// Если мы знаем калибровки камер (матрицы K0, K1 в структуре матриц P0, P1), то можем наложить дополнительные ограничения, в частности, известно, что
// существенная матрица (Essential matrix = K1t * F * K0) имеет ровно два совпадающих ненулевых сингулярных значения, тогда как для фундаментальной матрицы они могут различаться
// Это дополнительное ограничение позволяет разложить существенную матрицу с точностью до 4 решений, вместо произвольного проективного преобразования (см. Hartley & Zisserman p.258)
// Обычно мы можем использовать одну общую калибровку, более менее верную для большого количества реальных камер и с ее помощью выполнить
// первичное разложение существенной матрицы (а из него, взаимное расположение камер) для последующего уточнения методом нелинейной оптимизации
void phg::decomposeEMatrix(cv::Matx34d& P0, cv::Matx34d& P1, const cv::Matx33d& Ecv, const std::vector<cv::Vec2d>& m0, const std::vector<cv::Vec2d>& m1, const Calibration& calib0, const Calibration& calib1)
{
    if (m0.size() != m1.size()) {
        throw std::runtime_error("decomposeEMatrix : m0.size() != m1.size()");
    }

    // пиксельные координаты -> нормализованные (умножаем на inv(K))
    std::vector<cv::Point2f> pts0_norm, pts1_norm;
    pts0_norm.reserve(m0.size());
    pts1_norm.reserve(m1.size());
    for (size_t i = 0; i < m0.size(); ++i) {
        cv::Vec3d norm0 = calib0.unproject(m0[i]); // (x, y, 1) в нормализованных координатах
        cv::Vec3d norm1 = calib1.unproject(m1[i]);
        pts0_norm.emplace_back(static_cast<float>(norm0[0] / norm0[2]), static_cast<float>(norm0[1] / norm0[2]));
        pts1_norm.emplace_back(static_cast<float>(norm1[0] / norm1[2]), static_cast<float>(norm1[1] / norm1[2]));
    }

    // Преобразуем cv::Matx33d в cv::Mat (необходимо для cv::recoverPose)
    cv::Mat E(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            E.at<double>(i, j) = Ecv(i, j);

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts0_norm, pts1_norm, R, t);
    if (inliers == 0) {
        throw std::runtime_error("decomposeEMatrix : recoverPose returned no inliers");
    }

    // P0 = [I | 0]
    P0 = cv::Matx34d::eye();

    // P1 = [R | t]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P1(i, j) = R.at<double>(i, j);
        }
        P1(i, 3) = t.at<double>(i, 0);
    }

    std::cout << "decomposeEMatrix: recoverPose returned " << inliers << " inliers\n";
    std::cout << "P0: \n" << P0 << std::endl;
    std::cout << "P1: \n" << P1 << std::endl;
}

void phg::decomposeUndistortedPMatrix(cv::Matx33d &R, cv::Vec3d &O, const cv::Matx34d &P)
{
    R = P.get_minor<3, 3>(0, 0);

    cv::Matx31d O_mat = -R.t() * P.get_minor<3, 1>(0, 3);
    O(0) = O_mat(0);
    O(1) = O_mat(1);
    O(2) = O_mat(2);

    if (cv::determinant(R) < 0) {
        R *= -1;
    }
}

cv::Matx33d phg::composeEMatrixRT(const cv::Matx33d &R, const cv::Vec3d &T)
{
    return skew(T) * R;
}

cv::Matx34d phg::composeCameraMatrixRO(const cv::Matx33d &R, const cv::Vec3d &O)
{
    vector3d T = -R * O;
    return make34(R, T);
}
