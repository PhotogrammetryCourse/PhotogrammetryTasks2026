#include "sfm_utils.h"

#include <algorithm>
#include <stdexcept>


// pseudorandom number generator
uint64_t xorshift64(uint64_t *state)
{
    if (*state == 0) {
        *state = 1;
    }

    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

void phg::randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state)
{
    dst.clear();

    const int max_attempts = 1000;

    for (int i = 0; i < sample_size; ++i) {
        for (int k = 0; k < max_attempts; ++k) {
            int v = xorshift64(state) % max_id;
            if (dst.empty() || std::find(dst.begin(), dst.end(), v) == dst.end()) {
                dst.push_back(v);
                break;
            }
        }
        if (dst.size() < i + 1) {
            throw std::runtime_error("Failed to sample ids");
        }
    }
}

// проверяет, что расстояние от точки до линии меньше порога
bool phg::epipolarTest(const cv::Vec2d &pt0, const cv::Vec2d &pt1, const cv::Matx33d &F, double t)
{
    const cv::Vec3d x0(pt0[0], pt0[1], 1.0);
    const cv::Vec3d x1(pt1[0], pt1[1], 1.0);
    const cv::Vec3d l1 = F * x0;

    const double a = l1[0];
    const double b = l1[1];
    const double c = l1[2];
    const double denom = std::sqrt(a * a + b * b);
    if (denom == 0.0) {
        return false;
    }

    const double dist = std::abs(a * x1[0] + b * x1[1] + c) / denom;
    return dist < t;
}
