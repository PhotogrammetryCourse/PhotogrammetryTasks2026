#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    Eigen::MatrixXd A(2 * count, 4);

    for (int i = 0; i < count; ++i) {
        const cv::Matx34d &P = Ps[i];
        cv::Vec3d m = ms[i];
        double x = m[0], y = m[1], w = m[2];

        cv::Matx<double, 1, 4> P1 = P.row(0);
        cv::Matx<double, 1, 4> P2 = P.row(1);
        cv::Matx<double, 1, 4> P3 = P.row(2);

        A(2 * i, 0) = x * P3(0) - w * P1(0);
        A(2 * i, 1) = x * P3(1) - w * P1(1);
        A(2 * i, 2) = x * P3(2) - w * P1(2);
        A(2 * i, 3) = x * P3(3) - w * P1(3);

        A(2 * i + 1, 0) = y * P3(0) - w * P2(0);
        A(2 * i + 1, 1) = y * P3(1) - w * P2(1);
        A(2 * i + 1, 2) = y * P3(2) - w * P2(2);
        A(2 * i + 1, 3) = y * P3(3) - w * P2(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3);
    return cv::Vec4d(X(0), X(1), X(2), X(3));
}
