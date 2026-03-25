#include "triangulation.h"

#include "defines.h"
#include "iostream"
#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::MatrixXd A(count*2, 4);

    for (int i = 0; i < count; ++i) {
        const cv::Matx34d &P = Ps[i];
        const cv::Vec3d &m = ms[i];

        A.row(i * 2) << m[0] * P(2, 0) - P(0, 0),
                    m[0] * P(2, 1) - P(0, 1),
                    m[0] * P(2, 2) - P(0, 2),
                    m[0] * P(2, 3) - P(0, 3);


        A.row(i * 2 + 1) << m[1] * P(2, 0) - P(1, 0),
                        m[1] * P(2, 1) - P(1, 1),
                        m[1] * P(2, 2) - P(1, 2),
                        m[1] * P(2, 3) - P(1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3); 
    return cv::Vec4d(X(0), X(1), X(2), X(3));
}
