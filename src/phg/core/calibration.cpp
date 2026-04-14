#include <cmath>
#include <phg/sfm/defines.h>
#include "calibration.h"

const int NUM_ITER = 10;

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
    double x = point[0] / point[2];
    double y = point[1] / point[2];

    // TODO 11: добавьте учет радиальных искажений (k1_, k2_) (после деления на Z, но до умножения на f)
    double r = x * x + y * y;
    double L = 1.0 + k1_ * r + k2_ * r * r;
    x = L * x;
    y = L * y;

    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    x /= f_;
    y /= f_;

    // TODO 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)

    // деление на полином не является математически обратной функцией к тому полиному, который используется в project.

    double x_next = x;
    double y_next = y;
    for(int i = 0; i < NUM_ITER; ++i) {
        double r = x_next * x_next + y_next * y_next;
        double L = 1.0 + k1_ * r + k2_ * r * r;
        x_next = x / L;
        y_next = y / L;

        double delta = std::max(std::abs(x_next * L - x), std::abs(y_next * L - y));
        if (delta < 1e-10) {
            break;
        }
        x_next = x / L;
        y_next = y / L;
    }

    return cv::Vec3d(x_next, y_next, 1.0);
}
