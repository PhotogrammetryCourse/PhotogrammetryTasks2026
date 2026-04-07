#include <array>
#include <cassert>
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

constexpr double phg::Calibration::radial_distortion_at(double r2) const {
    return 1 + k1_ * r2 + k2_ * r2 * r2;
}

cv::Vec3d phg::Calibration::project(const cv::Vec3d &point) const
{
    double x = point[0] / point[2];
    double y = point[1] / point[2];

    // 11: добавьте учет радиальных искажений (k1_, k2_) (после деления на Z, но до умножения на f)
    double r2 = x*x + y*y;
    double radial = radial_distortion_at(r2);
    x *= radial;
    y *= radial;

    x *= f_;
    y *= f_;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

struct Poly5Res {

    Poly5Res(const std::array<double, 6> &cs) : cs(cs) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        T res = cs[0];
        T xp = x[0];
        for (int i = 1; i < cs.size(); ++i) {
            res += xp * cs[i];
            xp *= x[0];
        }
        residual[0] = res;
        return true;
    }

    private:
    std::array<double, 6> cs;
};

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    x /= f_;
    y /= f_;

    // 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)

    double phi = std::atan2(y, x);
    const double base_r = std::hypot(x, y);
    double r = base_r;
    for (int i = 0; i < 4; ++i) r = base_r - r * (radial_distortion_at(r * r) - 1.0);

    return cv::Vec3d(std::cos(phi) * r, std::sin(phi) * r, 1.0);
}
