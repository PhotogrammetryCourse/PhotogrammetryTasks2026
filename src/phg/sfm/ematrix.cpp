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
        Eigen::MatrixXd E;
        copy(Ecv, E);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd s = svd.singularValues();
        const double sigma = 0.5 * (s[0] + s[1]);
        Eigen::Vector3d corrected_s(sigma, sigma, 0.0);
        E = svd.matrixU() * corrected_s.asDiagonal() * svd.matrixV().transpose();

        copy(E, Ecv);
    }

}

cv::Matx33d phg::fmatrix2ematrix(const cv::Matx33d &F, const phg::Calibration &calib0, const phg::Calibration &calib1)
{
    matrix3d E = calib1.K().t() * F * calib0.K();
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
        vector3d ray0 = calib0.unproject(m0);
        vector3d ray1 = calib1.unproject(m1);

        vector3d ps[2] = {ray0, ray1};
        matrix34d Ps[2] = {P0, P1};

        vector4d Xh = phg::triangulatePoint(Ps, ps, 2);
        if (Xh[3] == 0) {
            return false;
        }
        Xh /= Xh[3];

        const vector3d x0 = P0 * Xh;
        const vector3d x1 = P1 * Xh;
        return x0[2] > 0.0 && x1[2] > 0.0;
    }
}

// Матрицы камер для фундаментальной матрицы определены с точностью до проективного преобразования
// То есть, можно исказить трехмерный мир (применив 4-мерную однородную матрицу), и одновременно поменять матрицы P0, P1 так, что проекции в пикселях не изменятся
// Если мы знаем калибровки камер (матрицы K0, K1 в структуре матриц P0, P1), то можем наложить дополнительные ограничения, в частности, известно, что
// существенная матрица (Essential matrix = K1t * F * K0) имеет ровно два совпадающих ненулевых сингулярных значения, тогда как для фундаментальной матрицы они могут различаться
// Это дополнительное ограничение позволяет разложить существенную матрицу с точностью до 4 решений, вместо произвольного проективного преобразования (см. Hartley & Zisserman p.258)
// Обычно мы можем использовать одну общую калибровку, более менее верную для большого количества реальных камер и с ее помощью выполнить
// первичное разложение существенной матрицы (а из него, взаимное расположение камер) для последующего уточнения методом нелинейной оптимизации
void phg::decomposeEMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &Ecv, const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, const Calibration &calib0, const Calibration &calib1)
{
    if (m0.size() != m1.size()) {
        throw std::runtime_error("decomposeEMatrix : m0.size() != m1.size()");
    }

    cv::Mat E_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            E_cv.at<double>(r, c) = Ecv(r, c);
        }
    }

    cv::Mat R0_cv, R1_cv, t_cv;
    cv::decomposeEssentialMat(E_cv, R0_cv, R1_cv, t_cv);

    Eigen::MatrixXd R0(3, 3), R1(3, 3);
    Eigen::VectorXd t(3);
    for (int r = 0; r < 3; ++r) {
        t[r] = t_cv.at<double>(r, 0);
        for (int c = 0; c < 3; ++c) {
            R0(r, c) = R0_cv.at<double>(r, c);
            R1(r, c) = R1_cv.at<double>(r, c);
        }
    }

    Eigen::VectorXd t_pos = t;
    Eigen::VectorXd t_neg = -t;

    P0 = matrix34d::eye();

    matrix34d P10 = composeP(R0, t_pos);
    matrix34d P11 = composeP(R0, t_neg);
    matrix34d P12 = composeP(R1, t_pos);
    matrix34d P13 = composeP(R1, t_neg);
    matrix34d P1s[4] = {P10, P11, P12, P13};

    int best_count = 0;
    int best_idx = -1;
    for (int i = 0; i < 4; ++i) {
        int count = 0;
        for (int j = 0; j < (int) m0.size(); ++j) {
            if (depthTest(m0[j], m1[j], calib0, calib1, P0, P1s[i])) {
                ++count;
            }
        }
        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    if (best_count == 0) {
        throw std::runtime_error("decomposeEMatrix : can't decompose ematrix");
    }

    P1 = P1s[best_idx];
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
