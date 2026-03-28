#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, 
// там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов

    // по 2 уравнения на камеру + 4 неизвестных X, Y, Z, W
    Eigen::MatrixXd A(count * 2, 4);

    // составление уравнений
    for (int i = 0; i < count; i++)
    {
        cv::Matx34d P = Ps[i];
        cv::Vec3d m = ms[i];

        for (int j = 0; j < 4; j++) 
        {
            A(i * 2, j) = m[0] * P(2, j) - P(0, j);
            A(i * 2 + 1, j) = m[1] * P(2, j) - P(1, j);
        }
    }

    // решаем систему через SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd res = svd.matrixV().col(3);
    double X = res(0);
    double Y = res(1);
    double Z = res(2);
    double W = res(3);

    return cv::Vec4d(X, Y, Z, W);
}
