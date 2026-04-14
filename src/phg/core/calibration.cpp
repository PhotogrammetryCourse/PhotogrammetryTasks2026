#include <cmath>
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

    double r2 = x * x + y * y;
    double distortion = 1.0 + k1_ * r2 + k2_ * r2 * r2;
    x *= distortion;
    y *= distortion;

    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double xd = pixel[0] - cx_ - width_ * 0.5;
    double yd = pixel[1] - cy_ - height_ * 0.5;

    xd /= f_;
    yd /= f_;

    double x = xd;
    double y = yd;
    for (int iter = 0; iter < 5; ++iter) {
        double r2 = x * x + y * y;
        double distortion = 1.0 + k1_ * r2 + k2_ * r2 * r2;
        x = xd / distortion;
        y = yd / distortion;
    }

    return cv::Vec3d(x, y, 1.0);
}