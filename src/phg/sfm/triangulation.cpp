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
    // каждая камера даёт 2 уравнения: x*(P[2]·X) - P[0]·X = 0, y*(P[2]·X) - P[1]·X = 0
    Eigen::MatrixXd A(2 * count, 4);

    for (int i = 0; i < count; ++i) {
        double x = ms[i][0], y = ms[i][1];
        for (int j = 0; j < 4; ++j) {
            A(2*i,     j) = x * Ps[i](2, j) - Ps[i](0, j);
            A(2*i + 1, j) = y * Ps[i](2, j) - Ps[i](1, j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3);

    return cv::Vec4d(X[0], X[1], X[2], X[3]);
}
