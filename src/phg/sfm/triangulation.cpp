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
    Eigen::MatrixXd A(4, 4);
    A.row(0) << ms[0][0] * Ps[0](2, 0) - Ps[0](0, 0), 
                ms[0][0] * Ps[0](2, 1) - Ps[0](0, 1), 
                ms[0][0] * Ps[0](2, 2) - Ps[0](0, 2), 
                ms[0][0] * Ps[0](2, 3) - Ps[0](0, 3);

    A.row(1) << ms[0][1] * Ps[0](2, 0) - Ps[0](1, 0),
                ms[0][1] * Ps[0](2, 1) - Ps[0](1, 1), 
                ms[0][1] * Ps[0](2, 2) - Ps[0](1, 2), 
                ms[0][1] * Ps[0](2, 3) - Ps[0](1, 3);

    A.row(2) << ms[1][0] * Ps[1](2, 0) - Ps[1](0, 0), 
                ms[1][0] * Ps[1](2, 1) - Ps[1](0, 1), 
                ms[1][0] * Ps[1](2, 2) - Ps[1](0, 2), 
                ms[1][0] * Ps[1](2, 3) - Ps[1](0, 3);

    A.row(3) << ms[1][1] * Ps[1](2, 0) - Ps[1](1, 0), 
                ms[1][1] * Ps[1](2, 1) - Ps[1](1, 1), 
                ms[1][1] * Ps[1](2, 2) - Ps[1](1, 2), 
                ms[1][1] * Ps[1](2, 3) - Ps[1](1, 3);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto null_space = svd.matrixV().col(3);

    return {null_space[0], null_space[1], null_space[2], null_space[3]};
}
