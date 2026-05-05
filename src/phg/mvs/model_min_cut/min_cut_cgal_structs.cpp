#include "min_cut_cgal_structs.h"

#include <libutils/rasserts.h>

vector3d from_cgal_point(cgal_point_t p) { return vector3d(p.x(), p.y(), p.z()); }

cgal_point_t to_cgal_point(vector3d p) { return cgal_point_t(p[0], p[1], p[2]); }

vertex_info_t::vertex_info_t(unsigned int camera_id, const vector3d& point, const cv::Vec3b& color, double radius)
    : color(color)
    , radius(radius)
    , position_sum(point)
    , color_sum(color[0], color[1], color[2])
    , observations_count(1)
{
    camera_ids.push_back(camera_id);
}

void vertex_info_t::merge(const vertex_info_t& that)
{
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }
    for (int i = 1; i < that.camera_ids.size(); ++i) {
        rassert(that.camera_ids[i - 1] < that.camera_ids[i], 23781274121021);
    }

    for (int i = 0; i < that.camera_ids.size(); ++i) {
        unsigned int ci = that.camera_ids[i];
        if (std::find(camera_ids.begin(), camera_ids.end(), ci) == camera_ids.end()) {
            camera_ids.push_back(ci);
        }
    }

    std::sort(camera_ids.begin(), camera_ids.end());
    radius = std::max(radius, that.radius);
    position_sum += that.position_sum;
    color_sum += that.color_sum;
    observations_count += that.observations_count;
    if (observations_count > 0) {
        color = averageColor();
    }
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }
}

vector3d vertex_info_t::averagePoint(const vector3d& fallback_point) const
{
    if (observations_count == 0)
        return fallback_point;
    return position_sum / (double)observations_count;
}

cv::Vec3b vertex_info_t::averageColor() const
{
    if (observations_count == 0)
        return color;
    cv::Vec3b avg;
    for (int c = 0; c < 3; ++c) {
        avg[c] = cv::saturate_cast<unsigned char>(color_sum[c] / (double)observations_count);
    }
    return avg;
}
