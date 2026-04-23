#include <phg/sfm/defines.h>
#include "calibration.h"


phg::Calibration::Calibration(int width, int height)
    : width_(width)
    , height_(height)
    , cx_(0)
    , cy_(0)
    , k1_(0)
    , k2_(0)
{
    // 50mm guess

    double diag_35mm = 36.0 * 36.0 + 24.0 * 24.0;
    double diag_pix = (double) width * (double) width + (double) height * (double) height;

    f_ = 50.0 * std::sqrt(diag_pix / diag_35mm);
}

cv::Matx33d phg::Calibration::K() const {
    return {f_, 0., cx_ + width_ * 0.5, 0., f_, cy_ + height_ * 0.5, 0., 0., 1.};
}

int phg::Calibration::width() const {
    return width_;
}

int phg::Calibration::height() const {
    return height_;
}

cv::Vec3d phg::Calibration::project(const cv::Vec3d &point) const
{
    // return K() * point; // original from task03
    double x = point[0] / point[2];
    double y = point[1] / point[2];

    // 11: добавьте учет радиальных искажений (k1_, k2_) (после деления на Z, но до умножения на f)
    double r_squared = x * x + y * y;
    double r_fourth = r_squared * r_squared;
    double radial_coef = 1. + k1_ * r_squared + k2_ * r_fourth;

    x *= radial_coef;
    y *= radial_coef;


    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    // K().inv() * vector3d(pixel[0], pixel[1], 1.0);
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    x /= f_;
    y /= f_;

    // 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)
    // потому что мы имеем дело с функцией 4 степени относительно x и y. 
    double x_u = x, y_u = y;
    // while (1) {
    for (size_t i = 0; i < 1000; i++) {
        double r_squared = x_u * x_u + y_u * y_u;
        double r_fourth = r_squared * r_squared;
        double dr = 1 + k1_ * r_squared + k2_ * r_fourth;
        x_u = x / dr;
        y_u = y / dr;
        
        if (std::abs(dr * x_u - x) + std::abs(dr * y_u - y) < 1e-10) {
            break;
        }
    }

    return cv::Vec3d(x_u, y_u, 1.0);
}
