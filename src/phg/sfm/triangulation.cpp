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
    // Ps is a camera matrix, ms is a normalized image point (after applying inverse of K),
    // count is number of views
    // For two cameras (P, and P2) and observed image points (x1, x2),
    // this results in 4 linear equations (2 from each image)

    int a_rows = count * 2; // number of equations
    int a_cols = 4;         // homogeneous coordinates of the point

    // A matrix construction
    Eigen::MatrixXd A(a_rows, a_cols);

    for (int i = 0; i < count; ++i) {
        const cv::Matx34d &P = Ps[i];
        const cv::Vec3d &m = ms[i];

        // First row of equations
        A.row(i * 2) << m[0] * P(2, 0) - P(0, 0),
                        m[0] * P(2, 1) - P(0, 1),
                        m[0] * P(2, 2) - P(0, 2),
                        m[0] * P(2, 3) - P(0, 3);

        // Second row of equations
        A.row(i * 2 + 1) << m[1] * P(2, 0) - P(1, 0),
                            m[1] * P(2, 1) - P(1, 1),
                            m[1] * P(2, 2) - P(1, 2),
                            m[1] * P(2, 3) - P(1, 3);
    }

    // Solve the homogeneous system using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd X = svd.matrixV().col(3); // Last column of V corresponds to the solution

    // Return the triangulated point in homogeneous coordinates

    return cv::Vec4d(X(0), X(1), X(2), X(3));
}
