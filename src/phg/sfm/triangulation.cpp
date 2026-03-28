#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixXd A(2 * count, 4);
    for (int i = 0; i < count; ++i) {
        const cv::Matx34d &P = Ps[i];
        const cv::Vec3d &m = ms[i];

        for (int j = 0; j < 4; ++j) {
            A(2 * i + 0, j) = m[0] * P(2, j) - m[2] * P(0, j);
            A(2 * i + 1, j) = m[1] * P(2, j) - m[2] * P(1, j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd X = svd.matrixV().col(3);
    return {X[0], X[1], X[2], X[3]};
}
