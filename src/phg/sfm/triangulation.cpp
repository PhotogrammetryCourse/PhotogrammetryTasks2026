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
        for (int j = 0; j < 4; ++j) {
            A(2 * i, j) = ms[i][0] * Ps[i](2, j) - Ps[i](0, j);
            A(2 * i + 1, j) = ms[i][1] * Ps[i](2, j) - Ps[i](1, j);
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd v = svd.matrixV().col(3);
    return {v(0), v(1), v(2), v(3)};
}
