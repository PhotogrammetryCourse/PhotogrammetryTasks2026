#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/SVD/JacobiSVD.h>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixXd A(count * 2, 4);

    for (size_t i = 0; i < count; i++)
    {
        cv::Matx34d P = Ps[i];
        cv::Vec3d m = ms[i];

        for (int j = 0; j < 4; j++) 
        {
            A(i * 2, j) = m[0] * P(2, j) - P(0, j);
            A(i * 2 + 1, j) = m[1] * P(2, j) - P(1, j);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd res = svd.matrixV().col(3);

    return cv::Vec4d(res(0), res(1), res(2), res(3));
}
