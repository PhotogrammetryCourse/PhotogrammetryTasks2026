#include "sfm_utils.h"

#include <algorithm>
#include <stdexcept>

// pseudorandom number generator
uint64_t xorshift64(uint64_t* state)
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

void phg::randomSample(std::vector<int>& dst, int max_id, int sample_size, uint64_t* state)
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
bool phg::epipolarTest(const cv::Vec2d& pt0, const cv::Vec2d& pt1, const cv::Matx33d& F, double t)
{
    cv::Matx31d pt0_(pt0[0], pt0[1], 1.0);
    cv::Matx13d pt1_(pt1[0], pt1[1], 1.0);

    cv::Vec2d ep_line((F * pt0_)(0), (F * pt0_)(1));
    double length = cv::norm(ep_line);
    double distance = (pt1_ * F * pt0_)(0);
    double norm_distance = std::abs(distance / length);

    return norm_distance < t;
}
