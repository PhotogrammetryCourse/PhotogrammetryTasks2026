#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d* Ps, const cv::Vec3d* ms, int count)
{
    Eigen::MatrixXd matA(2 * count, 4);

    for (int idx = 0; idx < count; ++idx) {
        const auto& curr_p = Ps[idx];
        const auto& curr_m = ms[idx];

        cv::Vec4d p_row0(curr_p(0, 0), curr_p(0, 1), curr_p(0, 2), curr_p(0, 3));
        cv::Vec4d p_row1(curr_p(1, 0), curr_p(1, 1), curr_p(1, 2), curr_p(1, 3));
        cv::Vec4d p_row2(curr_p(2, 0), curr_p(2, 1), curr_p(2, 2), curr_p(2, 3));

        cv::Vec4d r1 = curr_m[0] * p_row2 - p_row0;
        cv::Vec4d r2 = curr_m[1] * p_row2 - p_row1;

        for (int j = 0; j < 4; ++j) {
            matA(2 * idx, j) = r1[j];
            matA(2 * idx + 1, j) = r2[j];
        }
    }

    Eigen::BDCSVD<Eigen::MatrixXd> solver(matA, Eigen::ComputeFullV);
    Eigen::Vector4d result = solver.matrixV().rightCols<1>();

    return cv::Vec4d(result.data());
}
