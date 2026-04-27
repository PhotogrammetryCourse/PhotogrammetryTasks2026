#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixXd A(2 * count, 4);

    for (int i = 0; i < count; ++i) {
        double mx = ms[i][0], my = ms[i][1], mw = ms[i][2];
        for (int col = 0; col < 4; ++col) {
            A(2 * i,     col) = mx * Ps[i](2, col) - mw * Ps[i](0, col);
            A(2 * i + 1, col) = my * Ps[i](2, col) - mw * Ps[i](1, col);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3);

    return cv::Vec4d(X[0], X[1], X[2], X[3]);
}
