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
    int a_rows = 2 * count;
    int a_cols = 4;

    Eigen::MatrixXd A(a_rows, a_cols);

    for (int i = 0; i < count; ++i) {
        double x = ms[i][0];
        double y = ms[i][1];
        const cv::Matx34d& P = Ps[i];

        A.row(2 * i) << x * P(2, 0) - P(0, 0), x * P(2, 1) - P(0, 1), x * P(2, 2) - P(0, 2), x * P(2, 3) - P(0, 3);
        A.row(2 * i + 1) << y * P(2, 0) - P(1, 0), y * P(2, 1) - P(1, 1), y * P(2, 2) - P(1, 2), y * P(2, 3) - P(1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svda.matrixV().col(a_cols - 1);

    return cv::Vec4d(null_space[0], null_space[1], null_space[2], null_space[3]);
}
