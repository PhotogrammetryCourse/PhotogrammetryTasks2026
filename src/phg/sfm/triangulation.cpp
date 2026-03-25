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

    for (int i = 0; i < count; i++) {
        cv::Vec3d m = ms[i];
        cv::Matx34d P = Ps[i];

        for (int j = 0; j < 4; j++) {
            A(i * 2, j) = m[0] * P(2, j) - m[2] * P(0, j);
            A(i * 2 + 1, j) = m[1] * P(2, j) - m[2] * P(1, j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svdf(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::VectorXd V = svdf.matrixV().col(3);

    return {V[0], V[1], V[2], V[3]};
}
